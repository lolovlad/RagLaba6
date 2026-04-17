from sentence_transformers import SentenceTransformer

from src.domain.interfaces.i_embedding_service import IEmbeddingService


class SentenceTransformerEmbeddingService(IEmbeddingService):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def embed_passage(self, text: str) -> list[float]:
        return self._model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

    def vector_size(self) -> int:
        return self._model.get_sentence_embedding_dimension()
