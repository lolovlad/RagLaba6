from __future__ import annotations

import logging
import os
import re
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from config_logging import setup_logging
from rag_pipeline import FinancialRAG
from indexer import index_document_from_bytes, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

log = logging.getLogger(__name__)
_rag: FinancialRAG | None = None


def get_rag() -> FinancialRAG:
    if _rag is None:
        raise RuntimeError("RAG pipeline not initialized")
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("Запуск приложения: инициализация RAG")
    global _rag
    _rag = FinancialRAG()
    log.info("RAG инициализирован")
    yield
    _rag = None
    log.info("Приложение остановлено")


app = FastAPI(
    title="RAG Financial Report QA",
    description="Вопрос-ответ по годовому финансовому отчёту (10-K) на базе RAG",
    lifespan=lifespan,
)


class QARequest(BaseModel):
    question: str = Field(..., description="Вопрос по отчёту")


class SourceItem(BaseModel):
    page: int | None
    type: str
    text: str | None = None
    rank: int | None = None


class QAResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str = "ok"
    export_path: str | None = None


def _safe_filename(name: str) -> str:
    base = Path(name).stem
    base = re.sub(r"[^\w\-_.]", "_", base)[:100]
    return f"{base}.pdf" if base else "document.pdf"


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(..., description="PDF-файл отчёта (10-K)")):
    log.info("POST /documents/upload: filename=%s", file.filename)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Требуется файл с расширением .pdf")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой")
    filename = _safe_filename(file.filename)
    docs_path = Path(os.getenv("DOCS_PATH", "./docs"))
    docs_path.mkdir(parents=True, exist_ok=True)
    save_path = docs_path / filename
    save_path.write_bytes(content)
    export_dir = docs_path / "export" / Path(filename).stem
    try:
        chunks = await asyncio.to_thread(
            index_document_from_bytes,
            content,
            filename,
            use_gpu=False,
            qdrant_host=os.getenv("QDRANT_HOST", QDRANT_HOST),
            qdrant_port=int(os.getenv("QDRANT_PORT", str(QDRANT_PORT))),
            collection=os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION),
            export_dir=export_dir,
        )
    except Exception as e:
        log.exception("Ошибка индексации при загрузке: %s", e)
        raise HTTPException(status_code=500, detail=f"Ошибка индексации: {e}")
    export_path_str = str(export_dir)
    log.info("POST /documents/upload успешно: filename=%s, chunks_indexed=%s, export_path=%s", filename, chunks, export_path_str)
    return UploadResponse(filename=filename, chunks_indexed=chunks, status="ok", export_path=export_path_str)


@app.post("/qa", response_model=QAResponse)
def qa(request: QARequest) -> QAResponse:
    log.info("POST /qa: question=%s", request.question[:80] + "..." if len(request.question) > 80 else request.question)
    try:
        rag = get_rag()
        result = rag.ask(request.question)
        sources = [
            SourceItem(
                page=s.get("page"),
                type=s.get("type", "text"),
                text=s.get("text"),
                rank=s.get("rank"),
            )
            for s in result.get("sources", [])
        ]
        log.info("POST /qa успешно: answer_len=%s, sources=%s", len(result.get("answer", "")), len(sources))
        return QAResponse(
            answer=result.get("answer", ""),
            sources=sources,
        )
    except Exception as e:
        log.exception("POST /qa ошибка: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
