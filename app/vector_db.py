from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from .config import VectorDBConfig, get_settings
from .core.base import BaseVectorStore
from .models import Document
from .utils import get_logger

class VectorStoreClient(BaseVectorStore):
    def __init__(self, config: VectorDBConfig) -> None:
        self._logger = get_logger(__name__)
        self._config = config
        http_auth = None
        if config.username and config.password:
            http_auth = (config.username, config.password)
        if not config.hosts:
            self._logger.warning("Vector store hosts not configured; client cannot connect")
        try:
            self._client = Elasticsearch(hosts=config.hosts, basic_auth=http_auth)
        except Exception as exc:
            self._logger.error("Failed to create vector store client: %s", exc)
            raise

    def _index_exists(self, index: str) -> bool:
        exists = self._client.indices.exists(index=index)
        if hasattr(exists, "body"):
            return exists.body
        return bool(exists)

    def _get_settings_data(self) -> tuple[List[str], List[str], Dict[str, str]]:
        """Get commonly used settings data to eliminate duplication"""
        settings = get_settings()
        columns, filter_fields, field_types = self._get_settings_data()
        return columns, filter_fields, field_types

    def ensure_index(self, dims: int) -> None:
        index = self._config.index
        settings = get_settings()

        if self._index_exists(index):
            return
        columns, filter_fields, field_types = self._get_settings_data()

        # Default required fields
        props: Dict[str, Any] = {
            "id": {"type": "keyword"},
            "text": {"type": "text"},
            "table": {"type": "keyword"},
            "source_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_offset": {"type": "integer"},
            "chunk_unit": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": self._config.similarity,
            },
            "metadata": {"type": "object", "enabled": True},
        }

        def _add_field(field_name: str) -> None:
            if field_name in props:
                return
            explicit = (field_types.get(field_name) or "").lower()
            if explicit in {"keyword", "text", "integer", "float"}:
                if explicit == "text":
                    props[field_name] = {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    }
                else:
                    props[field_name] = {"type": explicit}
                return
            if field_name in filter_fields or field_name.endswith("_id"):
                props[field_name] = {"type": "keyword"}
                return
            if field_name.endswith("_min") or field_name.endswith("_max"):
                base = field_name[:-4]
                base_type = (field_types.get(base) or "").lower()
                props[field_name] = {"type": "integer" if base_type == "integer" else "float"}
                return
            props[field_name] = {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            }
        for col in columns:
            _add_field(col)
        for f in filter_fields:
            _add_field(f)
        mapping_key = getattr(settings.vector_db, "mapping_properties_key", "properties")
        mapping: Dict[str, Any] = {"mappings": {mapping_key: props}}
        if settings.vector_db.index_settings:
            mapping["settings"] = settings.vector_db.index_settings
        if not self._config.similarity:
            self._logger.warning("Vector DB similarity not set; relying on backend defaults")
        try:
            self._client.indices.create(index=index, **mapping)
        except Exception as exc:
            self._logger.error("Failed to create index '%s': %s", index, exc)
            raise

    def delete_index(self) -> None:
        index = self._config.index
        try:
            if self._index_exists(index):
                self._client.indices.delete(index=index)
        except Exception as exc:
            self._logger.warning("Failed to delete index '%s': %s", index, exc)

    def upsert(
        self, documents: List[Document], embeddings: List[List[float]], refresh: Optional[bool] = None
    ) -> Tuple[int, int]:
        index = self._config.index
        settings = get_settings()
        columns: List[str] = list(settings.database.columns)
        filter_fields: List[str] = list(settings.retrieval.filter_fields)
        expected_dims = int(self._config.dims) if self._config.dims else None
        skipped_for_dim_mismatch = 0
        actions: List[Dict[str, Any]] = []
        for doc, vec in zip(documents, embeddings):
            if expected_dims and len(vec) != expected_dims:
                self._logger.error(
                    "Embedding dims mismatch for doc %s: got=%s expected=%s",
                    doc.id,
                    len(vec),
                    expected_dims,
                )
                skipped_for_dim_mismatch += 1
                continue
            source: Dict[str, Any] = {
                "id": doc.id,
                "text": doc.text,
                "embedding": vec,
                "metadata": doc.metadata,
            }
            # Add essential fields for indexing
            essential_fields = ["table", "source_id", "chunk_index", "chunk_offset", "chunk_unit"]
            for field in essential_fields:
                value = doc.metadata.get(field)
                if value is not None:
                    source[field] = value
            
            # Add only the most frequently used filter fields at root level for RAG performance
            # These are the fields you'll filter on most often in your RAG queries
            high_frequency_filters = [
                "emirate", "city", "property_type_id", "rent_type_id", 
                "furnishing_status", "number_of_bedrooms", "number_of_bathrooms",
                "property_status", "bnb_verification_status", "premiumBoostingStatus", 
                "carouselBoostingStatus", "rent_charge", "property_size"
            ]
            for field in high_frequency_filters:
                if field in source:
                    continue
                value = doc.metadata.get(field)
                if value is not None:
                    source[field] = value
            actions.append({"_op_type": "index", "_index": index, "_id": doc.id, "_source": source})
        try:
            success, errors = bulk(
                self._client,
                actions,
                refresh=refresh if refresh is not None else self._config.refresh_on_write,
            )
        except BulkIndexError as exc:
            error_items = exc.errors if hasattr(exc, "errors") else []
            self._logger.error("Bulk upsert failed: %s", exc)
            for err in error_items[:5]:
                self._logger.error("Bulk error item: %s", err)
            return 0, len(error_items)
        except Exception as exc:
            self._logger.error("Bulk upsert failed (unexpected): %s", exc)
            raise
        if isinstance(errors, list) and errors:
            for err in errors[:3]:
                self._logger.error("Bulk error item: %s", err)

        total_errors = (len(errors) if isinstance(errors, list) else 0) + skipped_for_dim_mismatch
        return success, total_errors

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        num_candidates: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        index = self._config.index
        settings = get_settings()
        knn = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(num_candidates, top_k),
        }
        es_query: Dict[str, Any] = {"knn": knn, "size": top_k}
        filter_clauses: List[Dict[str, Any]] = []
        if filters:
            allowed = set(settings.retrieval.filter_fields or [])
            # High-frequency filter fields that are stored at root level for performance
            high_frequency_filters = {
                "emirate", "city", "property_type_id", "rent_type_id", 
                "furnishing_status", "number_of_bedrooms", "number_of_bathrooms",
                "property_status", "bnb_verification_status", "premiumBoostingStatus", 
                "carouselBoostingStatus", "rent_charge", "property_size"
            }
            
            for key, value in filters.items():
                if key not in allowed or value is None:
                    continue
                
                # Use root-level field for high-frequency filters (faster)
                # Use metadata field for other filters (to avoid duplication)
                filter_key = key if key in high_frequency_filters else f"metadata.{key}"
                
                if isinstance(value, dict):
                    range_clause: Dict[str, Any] = {"range": {filter_key: {}}}
                    for bound in ("gte", "lte", "gt", "lt"):
                        if bound in value:
                            range_clause["range"][filter_key][bound] = value[bound]
                    if range_clause["range"][filter_key]:
                        filter_clauses.append(range_clause)
                elif isinstance(value, list):
                    filter_clauses.append({"terms": {filter_key: value}})
                else:
                    filter_clauses.append({"term": {filter_key: value}})
        if filter_clauses:
            es_query["query"] = {"bool": {"filter": filter_clauses}}

        try:
            resp = self._client.search(index=index, body=es_query, _source=True)
        except Exception as exc:
            self._logger.error("Vector store search request failed: %s", exc)
            raise
        hits = resp.get("hits", {}).get("hits", [])
        return hits

    def delete_by_source_id(self, source_id: str) -> int:
        index = self._config.index
        # Delete documents associated to the property id via source_id only
        query = {
            "bool": {
                "should": [
                    {"term": {"source_id": source_id}},
                    {"term": {"metadata.source_id": source_id}},
                ],
                "minimum_should_match": 1,
            }
        }
        try:
            resp = self._client.delete_by_query(index=index, body={"query": query}, refresh=self._config.refresh_on_write, conflicts="proceed")
        except Exception as exc:
            self._logger.error("Failed to delete by source_id %s: %s", source_id, exc)
            raise
        deleted = int(resp.get("deleted", 0)) if isinstance(resp, dict) else 0
        return deleted