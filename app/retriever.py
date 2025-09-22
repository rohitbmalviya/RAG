from __future__ import annotations
from typing import Any, Dict, List, Optional
from .config import RetrievalConfig
from .models import RetrievedChunk
from .core.base import BaseEmbedder, BaseVectorStore
from .utils import get_logger


ACTIVE_STATUS = "Active"
VERIFIED_STATUS = "verified"

class Retriever:
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore, config: RetrievalConfig) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config
        self._logger = get_logger(__name__)

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        print(f"🔍 RETRIEVER DEBUG:")
        print(f"   Query: '{query}'")
        print(f"   Raw filters: {filters}")
        
        if top_k is None and (self._config.top_k in (None, 0)):
            self._logger.warning("Retrieval top_k not provided and config.top_k is missing/zero; defaulting to 5")
        k = top_k if top_k is not None else (self._config.top_k or 5)
        if self._config.num_candidates_multiplier in (None, 0):
            self._logger.warning("Retrieval num_candidates_multiplier is missing/zero; using 1x top_k")
        num_candidates = k * max(1, self._config.num_candidates_multiplier or 1)
        query_vec = self._embedder.embed([query], task_type="retrieval_query")[0]
        
        # Validate filters against allowed fields
        validated_filters: Optional[Dict[str, Any]] = None
        if filters:
            allowed = set(self._config.filter_fields or [])
            validated_filters = {}
            for key, value in filters.items():
                if key in allowed:
                    validated_filters[key] = value
                else:
                    self._logger.warning("Ignoring invalid filter key '%s'. Allowed: %s", key, sorted(list(allowed)))
        
        print(f"   Validated filters: {validated_filters}")
        print(f"   Allowed filter fields: {sorted(list(self._config.filter_fields or []))}")
        
        try:
            hits = self._store.search(query_vec, top_k=k, num_candidates=num_candidates, filters=validated_filters)
            print(f"   Found {len(hits)} hits from vector store")
        except Exception as exc:
            self._logger.error("Vector store search failed: %s", exc)
            raise
        
        # Simplified scoring - let LLM handle the intelligence
        chunks: List[RetrievedChunk] = []
        for hit in hits:
            raw_score = float(hit.get("_score", 0.0))
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            
            # Simple boost for verified/premium properties
            boost_score = self._calculate_simple_boost(metadata)
            final_score = min(1.0, raw_score + boost_score)
            
            chunks.append(
                RetrievedChunk(score=final_score, text=source.get("text", ""), metadata=metadata)
            )
        
        # Sort by score (highest first)
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        print(f"   📊 FINAL RESULTS:")
        for i, chunk in enumerate(chunks[:5], 1):  
            metadata = chunk.metadata
            emirate = metadata.get("emirate", "Unknown")
            property_type = metadata.get("property_type_name", "Unknown")
            property_title = metadata.get("property_title", "Unknown")
            rent_charge = metadata.get("rent_charge", "Unknown")
            print(f"      {i}. {property_title} | Type: {property_type} | {emirate} | AED {rent_charge} | Score: {chunk.score:.3f}")
        
        return chunks

# Removed complex scoring methods - now using simple boost approach

    def _calculate_simple_boost(self, metadata: Dict[str, Any]) -> float:
        """Simple boost for verified/premium properties - let LLM handle the intelligence"""
        boost = 0.0
        
        # Small boost for verified properties
        if metadata.get("bnb_verification_status") == "verified":
            boost += 0.1
        
        # Small boost for premium properties
        if metadata.get("premiumBoostingStatus") == "Active":
            boost += 0.1
            
        # Small boost for carousel properties
        if metadata.get("carouselBoostingStatus") == "Active":
            boost += 0.05
        
        return min(0.2, boost)  # Cap total boost at 0.2  
