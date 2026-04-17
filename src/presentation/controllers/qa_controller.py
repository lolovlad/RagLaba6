from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.application.dtos.qa_dto import QARequest, QAResponse, UploadResponse
from src.application.interfaces.i_rag_service import IRagService
from src.application.services.document_indexing_service import DocumentIndexingService
from src.container import container

router = APIRouter()


def get_rag_service() -> IRagService:
    return container.resolve(IRagService)


def get_indexing_service() -> DocumentIndexingService:
    return container.resolve(DocumentIndexingService)


@router.post("/qa", response_model=QAResponse)
def answer_question(
    request: QARequest,
    rag_service: IRagService = Depends(get_rag_service),
) -> QAResponse:
    return rag_service.answer_question(request)


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    indexing_service: DocumentIndexingService = Depends(get_indexing_service),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not file.filename.lower().endswith((".pdf", ".html", ".htm")):
        raise HTTPException(status_code=400, detail="Only PDF or HTML files are supported")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")
    try:
        chunks = indexing_service.index_document(content=content, filename=file.filename)
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Docling parser dependencies are not installed. "
                "Run: poetry install."
            ),
        ) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {exc}") from exc
    return UploadResponse(filename=file.filename, chunks_indexed=chunks, status="ok")
