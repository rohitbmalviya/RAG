from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import RetrievalConfig
from .embedding import EmbeddingClient
from .models import RetrievedChunk
from .vector_db import ElasticsearchVectorStore
from .utils import get_logger


class Retriever:
    def __init__(self, embedder: EmbeddingClient, store: ElasticsearchVectorStore, config: RetrievalConfig) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config
        self._logger = get_logger(__name__)

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k if top_k is not None else self._config.top_k
        num_candidates = k * max(1, self._config.num_candidates_multiplier)
        query_vec = self._embedder.embed_texts([query], task_type="retrieval_query")[0]

        validated_filters: Optional[Dict[str, Any]] = None
        if filters:
            allowed = set(self._config.filter_fields or [])
            validated_filters = {}
            for key, value in filters.items():
                if key in allowed:
                    validated_filters[key] = value
                else:
                    self._logger.warning("Ignoring invalid filter key '%s'. Allowed: %s", key, sorted(list(allowed)))

        try:
            hits = self._store.search(query_vec, top_k=k, num_candidates=num_candidates, filters=validated_filters)
        except Exception as exc:
            self._logger.error("Elasticsearch search failed: %s", exc)
            raise
        chunks: List[RetrievedChunk] = []
        raw_scores: List[float] = [float(h.get("_score", 0.0)) for h in hits]
        max_score = max(raw_scores) if raw_scores else 1.0
        min_score = min(raw_scores) if raw_scores else 0.0
        denom = (max_score - min_score) if (max_score - min_score) > 0 else 1.0
        for hit in hits:
            raw = float(hit.get("_score", 0.0))
            score = (raw - min_score) / denom
            source = hit.get("_source", {})
            chunks.append(
                RetrievedChunk(score=score, text=source.get("text", ""), metadata=source.get("metadata", {}))
            )
        return chunks
