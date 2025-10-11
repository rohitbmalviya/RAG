from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from .config import VectorDBConfig, get_settings
from .core.base import BaseVectorStore
from .models import Document
from .utils import get_logger

KEYWORD_TYPE = "keyword"
TEXT_TYPE = "text"
INTEGER_TYPE = "integer"
FLOAT_TYPE = "float"

# All filters are now handled dynamically through metadata fields

class VectorStoreClient(BaseVectorStore):
    def __init__(self, config: VectorDBConfig) -> None:
        self._logger = get_logger(__name__)
        self._config = config
        http_auth = None
        if config.username and config.password:
            http_auth = (config.username, config.password)
        if not config.hosts:
            self._logger.debug("Vector store hosts not configured; client cannot connect")
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
    
    def _build_filter_key(self, field_name: str, field_types: Dict[str, str]) -> str:
        """
        Build the correct Elasticsearch filter key based on field type from config.
        
        For text/keyword fields stored in metadata, they have TEXT type with .keyword subfield.
        We need to use .keyword for exact matching.
        
        Args:
            field_name: The field name (e.g., 'emirate', 'property_type_name')
            field_types: Dictionary from config.yaml field_types section
            
        Returns:
            Elasticsearch field path (e.g., 'metadata.emirate.keyword')
        """
        field_type = field_types.get(field_name, "").lower()
        
        # Base path (all filters go through metadata in our schema)
        filter_key = f"metadata.{field_name}"
        
        # For text/keyword fields in metadata, use .keyword subfield for exact matching
        # Numeric, boolean, and date fields don't need .keyword
        if field_type in ["keyword", "text"]:
            filter_key = f"{filter_key}.keyword"
            self._logger.debug(f"🔍 VECTOR_DB: Field '{field_name}' is {field_type} → Using .keyword suffix")
        else:
            self._logger.debug(f"🔍 VECTOR_DB: Field '{field_name}' is {field_type} → Using as-is")
        
        return filter_key

    def _get_settings_data(self) -> tuple[List[str], List[str], Dict[str, str]]:
        """Get commonly used settings data to eliminate duplication"""
        settings = get_settings()
        columns = list(settings.database.columns)
        filter_fields = list(settings.retrieval.filter_fields)
        field_types = settings.database.field_types or {}
        return columns, filter_fields, field_types

    def _get_index_name(self) -> str:
        """Get index name to eliminate duplication"""
        return self._config.index

    def _process_metadata_fields(self, doc: Document, source: Dict[str, Any], fields: List[str]) -> None:
        """Process metadata fields and add to source if not None"""
        for field in fields:
            if field in source:
                continue
            value = doc.metadata.get(field)
            if value is not None:
                source[field] = value

    def _log_bulk_errors(self, errors: List[Any], max_errors: int = 5) -> None:
        """Log bulk operation errors with limit"""
        for err in errors[:max_errors]:
            self._logger.error("Bulk error item: %s", err)

    def ensure_index(self, dims: int) -> None:
        index = self._get_index_name()
        settings = get_settings()

        if self._index_exists(index):
            return
        columns, filter_fields, field_types = self._get_settings_data()

        props: Dict[str, Any] = {
            "id": {"type": KEYWORD_TYPE},
            "text": {"type": TEXT_TYPE},
            "table": {"type": KEYWORD_TYPE},
            "source_id": {"type": KEYWORD_TYPE},
            "chunk_index": {"type": INTEGER_TYPE},
            "chunk_offset": {"type": INTEGER_TYPE},
            "chunk_unit": {"type": KEYWORD_TYPE},
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": self._config.similarity
            },
            "metadata": {"type": "object", "enabled": True},
        }

        def _add_field(field_name: str) -> None:
            if field_name in props:
                return
            explicit = (field_types.get(field_name) or "").lower()
            if explicit in {KEYWORD_TYPE, TEXT_TYPE, INTEGER_TYPE, FLOAT_TYPE}:
                if explicit == TEXT_TYPE:
                    props[field_name] = {
                        "type": TEXT_TYPE,
                        "fields": {"keyword": {"type": KEYWORD_TYPE, "ignore_above": 256}},
                    }
                else:
                    props[field_name] = {"type": explicit}
                return
            if field_name in filter_fields or field_name.endswith("_id"):
                props[field_name] = {"type": KEYWORD_TYPE}
                return
            if field_name.endswith("_min") or field_name.endswith("_max"):
                base = field_name[:-4]
                base_type = (field_types.get(base) or "").lower()
                props[field_name] = {"type": INTEGER_TYPE if base_type == INTEGER_TYPE else FLOAT_TYPE}
                return
            props[field_name] = {
                "type": TEXT_TYPE,
                "fields": {"keyword": {"type": KEYWORD_TYPE, "ignore_above": 256}},
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
            self._logger.debug("Vector DB similarity not set; relying on backend defaults")
        try:
            self._client.indices.create(index=index, **mapping)
        except Exception as exc:
            self._logger.error("Failed to create index '%s': %s", index, exc)
            raise

    def delete_index(self) -> None:
        index = self._get_index_name()
        try:
            if self._index_exists(index):
                self._client.indices.delete(index=index)
        except Exception as exc:
            self._logger.debug("Failed to delete index '%s': %s", index, exc)

    def upsert(
        self, documents: List[Document], embeddings: List[List[float]], refresh: Optional[bool] = None
    ) -> Tuple[int, int]:
        index = self._get_index_name()
        columns, filter_fields, field_types = self._get_settings_data()
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
            
            essential_fields = ["table", "source_id", "chunk_index", "chunk_offset", "chunk_unit"]
            self._process_metadata_fields(doc, source, essential_fields)
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
            self._log_bulk_errors(error_items, 5)
            return 0, len(error_items)
        except Exception as exc:
            self._logger.error("Bulk upsert failed (unexpected): %s", exc)
            raise
        if isinstance(errors, list) and errors:
            self._log_bulk_errors(errors, 3)

        total_errors = (len(errors) if isinstance(errors, list) else 0) + skipped_for_dim_mismatch
        return success, total_errors

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        num_candidates: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        index = self._get_index_name()
        settings = get_settings()
        
        self._logger.debug(f"🔍 VECTOR_DB: Starting vector search")
        self._logger.debug(f"🔍 VECTOR_DB: Index: {index}")
        self._logger.debug(f"🔍 VECTOR_DB: Top K: {top_k}")
        self._logger.debug(f"🔍 VECTOR_DB: Num Candidates: {num_candidates}")
        self._logger.debug(f"🔍 VECTOR_DB: Filters received: {filters}")
        self._logger.debug(f"🔍 VECTOR_DB: Query vector length: {len(query_vector)}")
        
        knn = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(num_candidates, top_k),
        }
        filter_clauses: List[Dict[str, Any]] = []
        
        if filters:
            allowed = set(settings.retrieval.filter_fields or [])
            self._logger.debug(f"🔍 VECTOR_DB: Allowed filter fields: {sorted(list(allowed))}")
            
            # Get field types from config for determining .keyword suffix
            field_types = settings.database.field_types or {}
            
            for key, value in filters.items():
                if key not in allowed or value is None:
                    self._logger.debug(f"🔍 VECTOR_DB: Skipping filter {key}: not allowed or None")
                    continue
                
                # Build filter key with .keyword suffix for text/keyword fields
                filter_key = self._build_filter_key(key, field_types)
                self._logger.debug(f"🔍 VECTOR_DB: Processing filter: {key} = {value} -> {filter_key}")
                if isinstance(value, dict):
                    # Handle range queries (gte, lte, gt, lt)
                    range_clause: Dict[str, Any] = {"range": {filter_key: {}}}
                    for bound in ("gte", "lte", "gt", "lt"):
                        if bound in value:
                            range_clause["range"][filter_key][bound] = value[bound]
                    if range_clause["range"][filter_key]:
                        filter_clauses.append(range_clause)
                        self._logger.debug(f"🔍 VECTOR_DB: Added range filter: {range_clause}")
                elif isinstance(value, list):
                    # Handle multiple values
                    filter_clauses.append({"terms": {filter_key: value}})
                    self._logger.debug(f"🔍 VECTOR_DB: Added terms filter: {filter_key} in {value}")
                else:
                    # Handle single value
                    filter_clauses.append({"term": {filter_key: value}})
                    self._logger.debug(f"🔍 VECTOR_DB: Added term filter: {filter_key} = {value}")
        
        # CRITICAL FIX: Put filters INSIDE kNN (not separate query)
        # This makes Elasticsearch pre-filter BEFORE doing vector similarity search
        if filter_clauses:
            # Simplify: single filter doesn't need bool wrapper
            if len(filter_clauses) == 1:
                knn["filter"] = filter_clauses[0]
                self._logger.debug(f"🔍 VECTOR_DB: Added single filter INSIDE kNN")
            else:
                knn["filter"] = {"bool": {"must": filter_clauses}}
                self._logger.debug(f"🔍 VECTOR_DB: Added multiple filters INSIDE kNN with bool/must")
            self._logger.debug(f"🔍 VECTOR_DB: kNN filter: {knn.get('filter')}")
        else:
            self._logger.debug(f"🔍 VECTOR_DB: No filters applied, using KNN only")
        
        es_query: Dict[str, Any] = {"knn": knn, "size": top_k}
        self._logger.debug(f"🔍 VECTOR_DB: Final ES query structure: knn with {len(filter_clauses)} filters")
            
        try:
            self._logger.debug(f"🔍 VECTOR_DB: Executing Elasticsearch query...")
            resp = self._client.search(index=index, body=es_query, _source=True)
            self._logger.debug(f"🔍 VECTOR_DB: Elasticsearch query completed successfully")
        except Exception as exc:
            self._logger.error(f"🔍 VECTOR_DB: Vector store search request failed: {exc}")
            raise
        
        hits = resp.get("hits", {}).get("hits", [])
        total_hits = resp.get("hits", {}).get("total", {}).get("value", 0)
        self._logger.debug(f"🔍 VECTOR_DB: Raw hits count: {len(hits)}")
        self._logger.debug(f"🔍 VECTOR_DB: Total matching documents: {total_hits}")
        
        # Log detailed information about each hit
        primary_field = settings.database.primary_display_field
        self._logger.debug(f"🔍 VECTOR_DB: Showing details for {min(len(hits), 10)} hits:")
        
        for i, hit in enumerate(hits[:10], 1):
            score = hit.get("_score", 0.0)
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            text = source.get("text", "")
            
            display_value = metadata.get(primary_field, "Unknown")
            doc_id = metadata.get("id", "N/A")
            
            self._logger.debug(f"🔍 VECTOR_DB: Hit {i}: {display_value} (ID: {doc_id}, Score: {score:.3f})")
            self._logger.debug(f"🔍 VECTOR_DB: Hit {i} full metadata: {metadata}")
            self._logger.debug(f"🔍 VECTOR_DB: Hit {i} text preview: {text[:200]}...")
            
            # Log key property details
            if metadata:
                self._logger.debug(f"🔍 VECTOR_DB: Hit {i} details: Rent={metadata.get('rent_charge', 'N/A')}, "
                                f"Bedrooms={metadata.get('number_of_bedrooms', 'N/A')}, "
                                f"Location={metadata.get('emirate', 'N/A')}, {metadata.get('city', 'N/A')}")
        
        return hits

    def delete_by_source_id(self, source_id: str) -> int:
        index = self._get_index_name()
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