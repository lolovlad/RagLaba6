import logging
import os

from src.domain.interfaces.i_document_ingestion_service import IDocumentIngestionService
from src.domain.interfaces.i_embedding_service import IEmbeddingService
from src.domain.interfaces.i_vector_store_service import IVectorStoreService

log = logging.getLogger(__name__)


class DocumentIndexingService:
    def __init__(
        self,
        ingestion_service: IDocumentIngestionService,
        embedding_service: IEmbeddingService,
        vector_store_service: IVectorStoreService,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._embedding_service = embedding_service
        self._vector_store_service = vector_store_service
        self._embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    def index_document(self, content: bytes, filename: str) -> int:
        log.info("[Indexing] Start document indexing: file=%s size_bytes=%s", filename, len(content))
        log.info("[Indexing] Stage 1/4: ensure Qdrant collection")
        self._vector_store_service.ensure_collection(vector_size=self._embedding_service.vector_size())
        log.info("[Indexing] Stage 2/4: parse and chunk document")
        chunks = self._ingestion_service.ingest(content=content, filename=filename)
        if not chunks:
            log.warning("[Indexing] No chunks extracted: file=%s", filename)
            raise ValueError("No chunks extracted from document")
        log.info("[Indexing] Chunks extracted: count=%s", len(chunks))
        log.info("[Indexing] Stage 3/4: generate embeddings")
        vectors: list[list[float]] = []
        total_batches = (len(chunks) + self._embedding_batch_size - 1) // self._embedding_batch_size
        for start in range(0, len(chunks), self._embedding_batch_size):
            batch = chunks[start : start + self._embedding_batch_size]
            batch_no = (start // self._embedding_batch_size) + 1
            log.info(
                "[Indexing] Embedding batch %s/%s: batch_size=%s",
                batch_no,
                total_batches,
                len(batch),
            )
            vectors.extend(self._embedding_service.embed_passage(chunk.text) for chunk in batch)
        log.info("[Indexing] Embeddings ready: count=%s", len(vectors))
        log.info("[Indexing] Stage 4/4: upsert to Qdrant")
        inserted = self._vector_store_service.index_chunks(chunks=chunks, vectors=vectors)
        log.info("[Indexing] Completed successfully: inserted=%s file=%s", inserted, filename)
        return inserted
