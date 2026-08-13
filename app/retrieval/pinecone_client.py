"""Pinecone adapter; all index operations use the fixed legal-docs namespace."""
from __future__ import annotations
from typing import Any
from pinecone import Pinecone, ServerlessSpec
from app.config import Settings


class VectorStoreUnavailable(RuntimeError): pass
class EmbeddingModelMismatch(RuntimeError): pass


class PineconeStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._index = None

    def _get_index(self):
        if not self.settings.pinecone_api_key:
            raise VectorStoreUnavailable("Pinecone API key is not configured")
        if self._index is None:
            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            names = [item["name"] if isinstance(item, dict) else item.name for item in pc.list_indexes()]
            if self.settings.pinecone_index_name not in names:
                pc.create_index(self.settings.pinecone_index_name, dimension=self.settings.embedding_dimension, metric="cosine", spec=ServerlessSpec(cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region))
            self._index = pc.Index(self.settings.pinecone_index_name)
        return self._index

    def upsert(self, vectors: list[dict[str, Any]]) -> None:
        self._get_index().upsert(vectors=vectors, namespace=self.settings.namespace)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete existing document vectors only when the namespace already exists.

        Pinecone creates namespaces on upsert, while deleting a namespace that has
        never received an upsert returns 404. A concurrent first upsert may still
        race this check, so only that specific 404 is treated as a no-op.
        """
        index = self._get_index()
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else getattr(stats, "namespaces", {})
        if self.settings.namespace not in namespaces:
            return
        try:
            index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=self.settings.namespace)
        except Exception as exc:
            status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
            if status == 404 or "404" in str(exc):
                return
            raise

    def query(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        response = self._get_index().query(vector=vector, top_k=top_k, include_metadata=True, namespace=self.settings.namespace)
        results = []
        # Support both dict-style and object-style responses from different Pinecone client versions
        matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", []) or []
        for match in matches:
            if isinstance(match, dict):
                metadata = dict(match.get("metadata") or {})
                match_id = match.get("id")
                score = match.get("score")
            else:
                metadata = dict(match.metadata or {})
                match_id = match.id
                score = getattr(match, "score", None)

            if metadata.get("embedding_model") != self.settings.embedding_model or metadata.get("embedding_revision") != self.settings.embedding_revision:
                raise EmbeddingModelMismatch("index built with a different embedding model")

            results.append({"chunk_id": match_id, "score": float(score) if score is not None else None, "metadata": metadata})
        return results
