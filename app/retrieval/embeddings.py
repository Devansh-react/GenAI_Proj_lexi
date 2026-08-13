"""Lazy Qwen embedding wrapper using the model's asymmetric encoders."""
from __future__ import annotations
from app.config import Settings


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.settings.embedding_model, revision=self.settings.embedding_revision, trust_remote_code=True)
        return self._model

    def encode_query(self, text: str) -> list[float]:
        model = self._load()
        # encode_query / encode_document were added in newer
        # sentence-transformers releases. Qwen exposes the equivalent query
        # prompt through the stable encode API, so older compatible releases
        # continue to work without changing the retrieval contract.
        if hasattr(model, "encode_query"):
            vector = model.encode_query(text, normalize_embeddings=True)
        else:
            vector = model.encode(text, prompt_name="query", normalize_embeddings=True)
        return vector.tolist()

    def encode_document(self, text: str) -> list[float]:
        model = self._load()
        if hasattr(model, "encode_document"):
            vector = model.encode_document(text, normalize_embeddings=True)
        else:
            # Qwen3 documents use the model's default (no query instruction).
            vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
