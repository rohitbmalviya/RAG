from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .config import VectorDBConfig
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
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "table": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "chunk_offset": {"type": "integer"},
                    "chunk_unit": {"type": "keyword"},
                    "property_title": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
                    "city": {"type": "keyword"},
                    "district": {"type": "keyword"},
                    "emirate": {"type": "keyword"},
                    "property_type_id": {"type": "keyword"},
                    "embedding": {"type": "dense_vector", "dims": dims, "index": True, "similarity": self._config.similarity},
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        }
        self._client.indices.create(index=index, mappings=mapping["mappings"])

    def upsert(self, documents: List[Document], embeddings: List[List[float]], refresh: Optional[bool] = None) -> Tuple[int, int]:
        index = self._config.index
        actions: List[Dict[str, Any]] = []
        for doc, vec in zip(documents, embeddings):
            source: Dict[str, Any] = {
                "id": doc.id,
                "text": doc.text,
                "embedding": vec,
                "metadata": doc.metadata,
                "table": doc.metadata.get("table"),
                "source_id": doc.metadata.get("source_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "chunk_offset": doc.metadata.get("chunk_offset"),
                "chunk_unit": doc.metadata.get("chunk_unit"),
                "property_title": doc.metadata.get("property_title"),
                "city": doc.metadata.get("city"),
                "district": doc.metadata.get("district"),
                "emirate": doc.metadata.get("emirate"),
                "property_type_id": str(doc.metadata.get("property_type_id")) if doc.metadata.get("property_type_id") is not None else None,
            }
            actions.append({
                "_op_type": "index",
                "_index": index,
                "_id": doc.id,
                "_source": source,
            })
        success, errors = bulk(self._client, actions, refresh=refresh if refresh is not None else self._config.refresh_on_write)
        return success, len(errors) if isinstance(errors, list) else 0

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
