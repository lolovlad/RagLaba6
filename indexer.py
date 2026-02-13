from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.parser import FinancialReportParser, DocumentBlock
from utils.chunker import iter_hybrid_chunks, CHUNKER_MAX_TOKENS

log = logging.getLogger(__name__)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
DOCS_PATH = os.getenv("DOCS_PATH", "./docs")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "


def get_embedding_model() -> SentenceTransformer:
    log.info("Загрузка модели эмбеддингов: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Модель эмбеддингов загружена, размерность=%s", model.get_sentence_embedding_dimension())
    return model


def create_collection(client: QdrantClient, vector_size: int, recreate: bool = False) -> None:
    collections = client.get_collections().collections
    if any(c.name == QDRANT_COLLECTION for c in collections):
        if recreate:
            log.info("Пересоздание коллекции: %s", QDRANT_COLLECTION)
            client.delete_collection(QDRANT_COLLECTION)
        else:
            log.info("Коллекция уже существует: %s", QDRANT_COLLECTION)
            return
    log.info("Создание коллекции: %s, vector_size=%s, distance=COSINE", QDRANT_COLLECTION, vector_size)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info("Коллекция создана: %s", QDRANT_COLLECTION)


def ensure_collection_exists(client: QdrantClient, vector_size: int) -> None:
    create_collection(client, vector_size=vector_size, recreate=False)


def index_document_from_bytes(
    content: bytes,
    filename: str,
    use_gpu: bool = False,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    collection: str | None = None,
    export_dir: Path | None = None,
) -> int:
    host = qdrant_host or QDRANT_HOST
    port = qdrant_port or QDRANT_PORT
    coll = collection or QDRANT_COLLECTION
    parser = FinancialReportParser(use_gpu=use_gpu)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    model = get_embedding_model()
    vector_size = model.get_sentence_embedding_dimension()
    log.info("Индексация документа через API: filename=%s, size=%s bytes", filename, len(content))
    client = QdrantClient(host=host, port=port, check_compatibility=False)
    ensure_collection_exists(client, vector_size=vector_size)

    conv = parser.parse_bytes(content, filename=filename)
    if export_dir is not None:
        parser.export_to_directory(conv, filename, Path(export_dir))
    points: list[PointStruct] = []
    try:
        for text, page_number, block_type in iter_hybrid_chunks(
            conv.document, source_file=filename, max_tokens=CHUNKER_MAX_TOKENS
        ):
            text_with_prefix = f"{PASSAGE_PREFIX}{text}"
            vector = model.encode(text_with_prefix, normalize_embeddings=True).tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "source_file": filename,
                "page_number": page_number,
                "type": block_type,
                "text": text,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        log.info("Чанкинг: гибридный (Docling), чанков=%s", len(points))
    except Exception as e:
        log.warning("Гибридный чанкинг недоступен (%s), fallback на RecursiveCharacterTextSplitter", e)
        blocks = parser.extract_blocks(conv, source_file=filename)
        for block in blocks:
            for chunk in splitter.split_text(block.content):
                if not chunk.strip():
                    continue
                text_with_prefix = f"{PASSAGE_PREFIX}{chunk}"
                vector = model.encode(text_with_prefix, normalize_embeddings=True).tolist()
                point_id = str(uuid.uuid4())
                payload = {
                    "source_file": block.source_file,
                    "page_number": block.page_number,
                    "type": block.block_type,
                    "text": chunk,
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        log.info("Чанкинг: fallback, чанков=%s", len(points))
    if points:
        log.info("Загрузка в Qdrant: collection=%s, points=%s", coll, len(points))
        client.upsert(collection_name=coll, points=points)
    log.info("Индексация завершена: filename=%s, chunks_indexed=%s", filename, len(points))
    return len(points)
