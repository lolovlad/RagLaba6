from sentence_transformers import SentenceTransformer

from src.domain.interfaces.i_embedding_service import IEmbeddingService


class SentenceTransformerEmbeddingService(IEmbeddingService):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def embed_passage(self, text: str) -> list[float]:
        return self._model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [f"passage: {t}" for t in texts]
        batch_size = min(64, max(1, len(prefixed)))
        vectors = self._model.encode(prefixed, normalize_embeddings=True, batch_size=batch_size)
        if hasattr(vectors, "ndim") and vectors.ndim == 1:
            return [vectors.tolist()]
        return [row.tolist() for row in vectors]

    def vector_size(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Embedding model returned no vector dimension")
        return int(dim)
