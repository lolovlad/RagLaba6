from fastapi import FastAPI

from config_logging import setup_logging
from src.presentation.controllers.qa_controller import router as qa_router

setup_logging()

app = FastAPI(
    title="RAG 10-K Financial Reports",
    description="Clean Architecture service for question answering on annual financial reports",
)
app.include_router(qa_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
