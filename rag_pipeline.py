from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.prompt_builder import build_qa_prompt

log = logging.getLogger(__name__)
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "financial_reports")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_MODEL_FALLBACK = os.getenv("YANDEX_MODEL_FALLBACK")
YANDEX_LLM_BASE_URL = os.getenv("YANDEX_LLM_BASE_URL", "https://llm.api.cloud.yandex.net/v1")

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
TOP_K = 5
TOP_K_RETRIEVE = 20
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _get_llm(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url=YANDEX_LLM_BASE_URL,
        api_key=YANDEX_API_KEY,
        default_headers={"x-folder-id": YANDEX_FOLDER_ID or ""},
        temperature=0.3,
        max_tokens=2000,
        request_timeout=120,
    )


def _yandex_completion(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    log.info("Вызов Yandex GPT (LangChain): prompt_len=%s, model=%s", len(prompt), model)
    messages: List = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    llm = _get_llm(model)
    response = llm.invoke(messages)
    answer = (response.content or "").strip()
    if not answer:
        log.warning("Yandex GPT: пустой ответ")
        return "В отчете отсутствует достаточная информация для точного ответа."
    log.info("Yandex GPT ответ получен: answer_len=%s", len(answer))
    return answer


def _load_reranker():
    try:
        from FlagEmbedding import FlagReranker
        return FlagReranker(RERANKER_MODEL, use_fp16=True)
    except Exception as e:
        log.warning("Реранкер %s недоступен: %s", RERANKER_MODEL, e)
        return None


class FinancialRAG:
    def __init__(
        self,
        qdrant_host: str = None,
        qdrant_port: int = None,
        collection: str = None,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = TOP_K,
        top_k_retrieve: int = TOP_K_RETRIEVE,
        use_reranker: bool = True,
    ):
        self._host = qdrant_host or QDRANT_HOST
        self._port = qdrant_port or int(QDRANT_PORT)
        self._collection = collection or QDRANT_COLLECTION
        self._top_k = top_k
        self._top_k_retrieve = top_k_retrieve
        self._use_reranker = use_reranker
        self._embedding_model_name = embedding_model
        self._model: Optional[SentenceTransformer] = None
        self._client: Optional[QdrantClient] = None
        self._reranker = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("RAG: загрузка модели эмбеддингов %s", self._embedding_model_name)
            self._model = SentenceTransformer(self._embedding_model_name)
            log.info("RAG: модель эмбеддингов загружена")
        return self._model

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            log.info("Подключение к Qdrant: host=%s, port=%s", self._host, self._port)
            self._client = QdrantClient(
                host=self._host, port=self._port, check_compatibility=False
            )
        return self._client

    def embed_query(self, question: str) -> list:
        log.info("RAG: эмбеддинг запроса, question_len=%s", len(question))
        text = f"{QUERY_PREFIX}{question}"
        model = self._get_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def _get_reranker(self):
        if self._reranker is None and self._use_reranker:
            self._reranker = _load_reranker()
        return self._reranker

    def retrieve_context(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        vector = self.embed_query(question)
        client = self._get_client()
        limit = self._top_k_retrieve if self._get_reranker() else self._top_k
        log.info("RAG: поиск в Qdrant, collection=%s, limit=%s", self._collection, limit)
        results = client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        points = getattr(results, "points", None) or getattr(results, "result", []) or []
        log.info("RAG: найдено кандидатов: %s", len(points))

        items = []
        for r in points:
            payload = r.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            items.append({
                "text": text,
                "page": payload.get("page_number"),
                "type": payload.get("type", "text"),
            })

        reranker = self._get_reranker()
        if reranker and len(items) > self._top_k:
            pairs = [[question, it["text"]] for it in items]
            try:
                scores = reranker.compute_score(pairs, normalize=True)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                for i, it in enumerate(items):
                    it["score"] = scores[i] if i < len(scores) else 0.0
                items.sort(key=lambda x: x["score"], reverse=True)
                items = items[: self._top_k]
                log.info("RAG: реранкинг выполнен, оставлено top_k=%s", len(items))
            except Exception as e:
                log.warning("Ошибка реранкинга, используем порядок Qdrant: %s", e)
                items = items[: self._top_k]
        elif len(items) > self._top_k:
            items = items[: self._top_k]

        chunks = [it["text"] for it in items]
        max_snippet_len = 500
        sources = []
        for rank, it in enumerate(items, start=1):
            snippet = (it["text"] or "").strip()
            if len(snippet) > max_snippet_len:
                snippet = snippet[: max_snippet_len] + "..."
            sources.append({
                "page": it["page"],
                "type": it["type"],
                "text": snippet,
                "rank": rank,
            })
        context = "\n\n---\n\n".join(chunks) if chunks else ""
        log.info("RAG: контекст собран, context_len=%s, sources=%s", len(context), len(sources))
        return context, sources

    def build_prompt(self, context: str, question: str) -> str:
        log.info("RAG: сборка промпта, context_len=%s", len(context))
        return build_qa_prompt(context=context, question=question)

    def generate_answer(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return _yandex_completion(prompt=prompt, system_prompt=system_prompt, model=YANDEX_MODEL_FALLBACK)

    def ask(self, question: str) -> Dict[str, Any]:
        log.info("RAG ask: вопрос=%s", question[:100] + "..." if len(question) > 100 else question)
        context, sources = self.retrieve_context(question)
        if not context.strip():
            log.warning("RAG ask: контекст пуст, возврат заглушки")
            return {
                "answer": "В отчете отсутствует достаточная информация для точного ответа.",
                "sources": [],
            }
        prompt = self.build_prompt(context, question)
        answer = self.generate_answer(prompt)
        log.info("RAG ask: ответ получен, answer_len=%s, sources=%s", len(answer), len(sources))
        return {"answer": answer, "sources": sources}  # sources: page, type, text, rank
