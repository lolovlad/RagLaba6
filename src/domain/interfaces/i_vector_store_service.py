from abc import ABC, abstractmethod

from src.domain.entities.document_chunk import DocumentChunk


class IVectorStoreService(ABC):
    @abstractmethod
    def ensure_collection(self, vector_size: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def index_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], limit: int) -> list[dict]:
        raise NotImplementedError
