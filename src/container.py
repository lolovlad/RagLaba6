import os

import punq
from dotenv import load_dotenv

from src.application.interfaces.i_rag_service import IRagService
from src.application.services.document_indexing_service import DocumentIndexingService
from src.application.services.rag_service import RagService
from src.domain.interfaces.i_document_ingestion_service import IDocumentIngestionService
from src.domain.interfaces.i_embedding_service import IEmbeddingService
from src.domain.interfaces.i_llm_service import ILlmService
from src.domain.interfaces.i_vector_store_service import IVectorStoreService
from src.infrastructure.services.document_ingestion_service import DoclingDocumentIngestionService
from src.infrastructure.services.embedding_service import SentenceTransformerEmbeddingService
from src.infrastructure.services.llm_service import YandexLlmService
from src.infrastructure.services.vector_store_service import QdrantVectorStoreService

load_dotenv()

container = punq.Container()
container.register(IEmbeddingService, SentenceTransformerEmbeddingService, scope=punq.Scope.singleton)
container.register(IVectorStoreService, QdrantVectorStoreService, scope=punq.Scope.singleton)
container.register(ILlmService, YandexLlmService, scope=punq.Scope.singleton)
container.register(
    IDocumentIngestionService,
    DoclingDocumentIngestionService,
    scope=punq.Scope.singleton,
    docs_path=os.getenv("DOCS_PATH", "./docs"),
)
container.register(IRagService, RagService, scope=punq.Scope.singleton)
container.register(DocumentIndexingService, DocumentIndexingService, scope=punq.Scope.singleton)
