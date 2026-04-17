from abc import ABC, abstractmethod


class IEmbeddingService(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_passage(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def vector_size(self) -> int:
        raise NotImplementedError
