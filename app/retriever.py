from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import RetrievalConfig
from .embedding import EmbeddingClient
from .models import RetrievedChunk
from .vector_db import ElasticsearchVectorStore


class Retriever:
    def __init__(self, embedder: EmbeddingClient, store: ElasticsearchVectorStore, config: RetrievalConfig) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k if top_k is not None else self._config.top_k
        num_candidates = k * max(1, self._config.num_candidates_multiplier)
        query_vec = self._embedder.embed_texts([query], task_type="retrieval_query")[0]
        hits = self._store.search(query_vec, top_k=k, num_candidates=num_candidates, filters=filters)
        chunks: List[RetrievedChunk] = []
        for hit in hits:
            score = hit.get("_score", 0.0)
            source = hit.get("_source", {})
            chunks.append(
                RetrievedChunk(score=score, text=source.get("text", ""), metadata=source.get("metadata", {}))
            )
        return chunks
