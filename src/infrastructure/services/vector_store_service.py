import os
import uuid
import logging

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.domain.entities.document_chunk import DocumentChunk
from src.domain.interfaces.i_vector_store_service import IVectorStoreService

log = logging.getLogger(__name__)


class QdrantVectorStoreService(IVectorStoreService):
    def __init__(self) -> None:
        load_dotenv()
        self._host = os.getenv("QDRANT_HOST", "localhost")
        self._port = int(os.getenv("QDRANT_PORT", "6333"))
        self._collection_name = os.getenv("QDRANT_COLLECTION", "financial_reports_10k")
        self._timeout = int(os.getenv("QDRANT_TIMEOUT", "120"))
        self._batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "64"))
        self._client = QdrantClient(
            host=self._host,
            port=self._port,
            timeout=self._timeout,
            check_compatibility=False,
        )
        log.info(
            "[Qdrant] Client initialized: host=%s port=%s collection=%s timeout=%s batch_size=%s",
            self._host,
            self._port,
            self._collection_name,
            self._timeout,
            self._batch_size,
        )

    def ensure_collection(self, vector_size: int) -> None:
        log.info("[Qdrant] Checking collection: name=%s", self._collection_name)
        try:
            collections = self._client.get_collections().collections
        except Exception as exc:
            raise ConnectionError(
                f"Qdrant is unavailable at {self._host}:{self._port}. Root cause: {exc}"
            ) from exc
        if any(collection.name == self._collection_name for collection in collections):
            log.info("[Qdrant] Collection already exists: name=%s", self._collection_name)
            return
        log.info("[Qdrant] Creating collection: name=%s vector_size=%s", self._collection_name, vector_size)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        log.info("[Qdrant] Collection created: name=%s", self._collection_name)

    def index_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=False):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number,
                        "chunk_type": chunk.chunk_type,
                        "text": chunk.text,
                    },
                )
            )
        if points:
            total_batches = (len(points) + self._batch_size - 1) // self._batch_size
            log.info("[Qdrant] Upsert started: points=%s batches=%s", len(points), total_batches)
            try:
                for start in range(0, len(points), self._batch_size):
                    batch = points[start : start + self._batch_size]
                    batch_no = (start // self._batch_size) + 1
                    log.info("[Qdrant] Upsert batch %s/%s: batch_size=%s", batch_no, total_batches, len(batch))
                    self._client.upsert(
                        collection_name=self._collection_name,
                        points=batch,
                        wait=True,
                    )
                log.info("[Qdrant] Upsert completed successfully: inserted=%s", len(points))
            except Exception as exc:
                raise ConnectionError(
                    f"Failed to push chunks to Qdrant at {self._host}:{self._port}. Root cause: {exc}"
                ) from exc
        return len(points)

    def search(self, query_vector: list[float], limit: int) -> list[dict]:
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = result.points if hasattr(result, "points") else []
        return [{"payload": p.payload, "score": p.score, "id": p.id} for p in points]
