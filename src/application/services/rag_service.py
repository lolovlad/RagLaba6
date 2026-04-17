import logging

from src.application.dtos.qa_dto import QARequest, QAResponse, SourceItem
from src.application.interfaces.i_rag_service import IRagService
from src.domain.interfaces.i_embedding_service import IEmbeddingService
from src.domain.interfaces.i_llm_service import ILlmService
from src.domain.interfaces.i_vector_store_service import IVectorStoreService

log = logging.getLogger(__name__)


class RagService(IRagService):
    def __init__(
        self,
        embedding_service: IEmbeddingService,
        vector_store_service: IVectorStoreService,
        llm_service: ILlmService,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store_service = vector_store_service
        self._llm_service = llm_service

    def answer_question(self, request: QARequest) -> QAResponse:
        query_vector = self._embedding_service.embed_query(request.question)
        hits = self._vector_store_service.search(query_vector=query_vector, limit=request.top_k)
        if not hits:
            return QAResponse(
                answer="В отчете отсутствует достаточная информация для точного ответа.",
                sources=[],
            )
        context = "\n\n---\n\n".join((hit.get("payload") or {}).get("text", "") for hit in hits)
        answer = self._llm_service.generate_answer(question=request.question, context=context)
        sources: list[SourceItem] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            snippet = (payload.get("text") or "").strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            sources.append(
                SourceItem(
                    page=payload.get("page_number"),
                    source_file=payload.get("source_file", "unknown"),
                    chunk_type=payload.get("chunk_type", "text"),
                    text=snippet,
                    score=hit.get("score"),
                )
            )
        log.info("Answer generated with %s sources", len(sources))
        return QAResponse(answer=answer, sources=sources)
