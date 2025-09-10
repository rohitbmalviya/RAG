from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.helpers import BulkIndexError

from .config import VectorDBConfig, get_settings
from .models import Document
from .utils import get_logger


class ElasticsearchVectorStore:
    def __init__(self, config: VectorDBConfig) -> None:
        self._logger = get_logger(__name__)
        self._config = config
        http_auth = None
        if config.username and config.password:
            http_auth = (config.username, config.password)
        self._client = Elasticsearch(hosts=config.hosts, basic_auth=http_auth)

    def _index_exists(self, index: str) -> bool:
        return self._client.indices.exists(index=index).body if hasattr(self._client.indices.exists(index=""), "body") else self._client.indices.exists(index=index)

    def ensure_index(self, dims: int) -> None:
        index = self._config.index
        try:
            exists = self._client.indices.exists(index=index)
            if isinstance(exists, dict):
                exists = bool(exists.get("value", False))
        except Exception:
            exists = self._client.indices.exists(index=index)
        if exists:
            return

        # Build dynamic mapping from config
        settings = get_settings()
        columns: List[str] = list(settings.database.columns)
        filter_fields: List[str] = list(settings.retrieval.filter_fields)

        props: Dict[str, Any] = {
            "id": {"type": "keyword"},
            "text": {"type": "text"},
            # Backward-compatible/common metadata
            "table": {"type": "keyword"},
            "source_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_offset": {"type": "integer"},
            "chunk_unit": {"type": "keyword"},
            # Embedding vector
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": self._config.similarity,
            },
            # Full metadata object if needed for debugging
            "metadata": {"type": "object", "enabled": True},
        }

        # Add dynamic fields from columns
        for col in columns:
            # Avoid redefining fields that are already included as common metadata
            if col in props:
                continue
            # If column is intended for filtering or ends with _id, map as keyword
            if col in filter_fields or col.endswith("_id"):
                props[col] = {"type": "keyword"}
            else:
                props[col] = {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                }

        mapping = {"mappings": {"properties": props}}
        self._client.indices.create(index=index, mappings=mapping["mappings"])

    def delete_index(self) -> None:
        """Deletes the configured index if it exists."""
        index = self._config.index
        try:
            if self._client.indices.exists(index=index):
                self._client.indices.delete(index=index)
        except Exception as exc:
            self._logger.warning("Failed to delete index '%s': %s", index, exc)

    def upsert(self, documents: List[Document], embeddings: List[List[float]], refresh: Optional[bool] = None) -> Tuple[int, int]:
        index = self._config.index
        actions: List[Dict[str, Any]] = []
        expected_dims = int(self._config.dims) if self._config.dims else None
        skipped_for_dim_mismatch = 0
        # Read dynamic columns from config
        settings = get_settings()
        columns: List[str] = list(settings.database.columns)
        for doc, vec in zip(documents, embeddings):
            # Validate embedding length if we know expected dims
            if expected_dims is not None and len(vec) != expected_dims:
                self._logger.error(
                    "Embedding dims mismatch for doc %s: got=%s expected=%s",
                    doc.id, len(vec), expected_dims,
                )
                skipped_for_dim_mismatch += 1
                continue
            source: Dict[str, Any] = {
                "id": doc.id,
                "text": doc.text,
                "embedding": vec,
                "metadata": doc.metadata,
                # Backward-compatible/common metadata
                "table": doc.metadata.get("table"),
                "source_id": doc.metadata.get("source_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "chunk_offset": doc.metadata.get("chunk_offset"),
                "chunk_unit": doc.metadata.get("chunk_unit"),
            }
            # Dynamically include all configured columns present in metadata
            for col in columns:
                if col in source:  # avoid overwriting common fields if names collide
                    continue
                value = doc.metadata.get(col)
                # Normalize types for keyword fields like *_id if needed (store as str)
                if value is not None and (col.endswith("_id")):
                    try:
                        value = str(value)
                    except Exception:
                        pass
                source[col] = value
            actions.append({
                "_op_type": "index",
                "_index": index,
                "_id": doc.id,
                "_source": source,
            })
        try:
            success, errors = bulk(
                self._client,
                actions,
                refresh=refresh if refresh is not None else self._config.refresh_on_write,
            )
        except BulkIndexError as exc:
            # Log a few representative errors to help diagnose mapping issues
            error_items = exc.errors if hasattr(exc, "errors") else []
            self._logger.error("Bulk upsert failed: %s", exc)
            for err in error_items[:5]:
                self._logger.error("Bulk error item: %s", err)
            # Return (0, num_errors) so caller can continue
            return 0, len(error_items)
        except Exception as exc:
            self._logger.error("Bulk upsert failed (unexpected): %s", exc)
            raise
        # Log a few errors for diagnostics
        if isinstance(errors, list) and errors:
            for err in errors[:3]:
                self._logger.error("Bulk error item: %s", err)
        # Include any locally skipped docs due to dim mismatch in error count
        total_errors = (len(errors) if isinstance(errors, list) else 0) + skipped_for_dim_mismatch
        return success, total_errors

    def search(self, query_vector: List[float], top_k: int, num_candidates: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        index = self._config.index
        knn = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(num_candidates, top_k),
        }
        es_query: Dict[str, Any] = {"knn": knn, "size": top_k}
        filter_clauses: List[Dict[str, Any]] = []
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    filter_clauses.append({"terms": {key: value}})
                else:
                    filter_clauses.append({"term": {key: value}})
        if filter_clauses:
            es_query["query"] = {"bool": {"filter": filter_clauses}}
        resp = self._client.search(index=index, body=es_query, _source=True)
        hits = resp.get("hits", {}).get("hits", [])
        return hits
