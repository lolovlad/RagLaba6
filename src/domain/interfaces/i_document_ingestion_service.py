from abc import ABC, abstractmethod

from src.domain.entities.document_chunk import DocumentChunk


class IDocumentIngestionService(ABC):
    @abstractmethod
    def ingest(self, content: bytes, filename: str) -> list[DocumentChunk]:
        raise NotImplementedError
