from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question about 10-K report")
    top_k: int = Field(default=5, ge=1, le=15)


class SourceItem(BaseModel):
    page: int | None = None
    source_file: str
    chunk_type: str
    text: str
    score: float | None = None


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str = "ok"
