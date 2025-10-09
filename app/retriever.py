from __future__ import annotations
from typing import Any, Dict, List, Optional
from .config import RetrievalConfig
from .models import RetrievedChunk
from .core.base import BaseEmbedder, BaseVectorStore
from .utils import get_logger


class Retriever:
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore, config: RetrievalConfig) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config
        self._logger = get_logger(__name__)
        # Cache settings to avoid repeated get_settings() calls (optimization)
        from .config import get_settings
        self._settings = get_settings()

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
        
        # Apply prioritized context retrieval strategy
        prioritized_filters = self._apply_prioritized_retrieval_strategy(filters, query)
        
        # Validate filters against allowed fields
        validated_filters: Optional[Dict[str, Any]] = None
        if prioritized_filters:
            allowed = set(self._config.filter_fields or [])
            validated_filters = {}
            for key, value in prioritized_filters.items():
                if key in allowed:
                    validated_filters[key] = value
                else:
                    self._logger.warning("Ignoring invalid filter key '%s'. Allowed: %s", key, sorted(list(allowed)))
        
        print(f"   Prioritized filters: {prioritized_filters}")
        print(f"   Validated filters: {validated_filters}")
        print(f"   Allowed filter fields: {sorted(list(self._config.filter_fields or []))}")
        
        try:
            hits = self._store.search(query_vec, top_k=k, num_candidates=num_candidates, filters=validated_filters)
            print(f"   Found {len(hits)} hits from vector store")
        except Exception as exc:
            self._logger.error("Vector store search failed: %s", exc)
            raise
        
        # Enhanced scoring with prioritized context retrieval
        chunks: List[RetrievedChunk] = []
        for hit in hits:
            raw_score = float(hit.get("_score", 0.0))
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            
            # Enhanced boost for verified/premium properties with priority scoring
            boost_score = self._calculate_prioritized_boost(metadata, query)
            final_score = min(1.0, raw_score + boost_score)
            
            chunks.append(
                RetrievedChunk(score=final_score, text=source.get("text", ""), metadata=metadata)
            )
        
        # Apply post-retrieval prioritization for "best property" queries
        if self._is_best_property_query(query):
            chunks = self._prioritize_best_properties(chunks)
        
        # Sort by score (highest first)
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        # Generic debug output (uses config!)
        primary_field = self._settings.database.primary_display_field
        pricing_field = self._settings.database.pricing_field
        currency = self._settings.database.display.currency
        
        print(f"   📊 FINAL RESULTS:")
        for i, chunk in enumerate(chunks[:5], 1):  
            metadata = chunk.metadata
            display_value = metadata.get(primary_field, "Unknown")
            price_value = metadata.get(pricing_field, "N/A") if pricing_field else "N/A"
            print(f"      {i}. {display_value} | {currency} {price_value} | Score: {chunk.score:.3f}")
        
        return chunks

# Removed complex scoring methods - now using simple boost approach

    def _apply_prioritized_retrieval_strategy(self, filters: Optional[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
        """Apply prioritized context retrieval strategy based on config.
        GENERIC - uses filter_priority_groups from config!
        """
        if not filters:
            return filters
        
        # Get priority groups from config (GENERIC!)
        priority_groups = self._settings.database.filter_priority_groups
        
        if not priority_groups:
            # No priority configuration - return filters as is
            return filters
        
        prioritized_filters = {}
        
        # Apply filters in priority order based on config
        # Common priority order: location, budget, property_type, rooms, furnishing, lease, availability
        priority_order = ["location", "budget", "property_type", "rooms", "furnishing", "lease", "availability"]
        
        for group_name in priority_order:
            if group_name not in priority_groups:
                continue
            group_fields = priority_groups[group_name]
            for field in group_fields:
                if field in filters:
                    prioritized_filters[field] = filters[field]
        
        # Add priority features from config (medium priority)
        priority_fields = self._settings.database.priority_features or []
        for field in priority_fields:
            if field in filters and field not in prioritized_filters:
                prioritized_filters[field] = filters[field]
        
        # Add any remaining filters not covered above
        for key, value in filters.items():
            if key not in prioritized_filters:
                prioritized_filters[key] = value
        
        return prioritized_filters

    def _is_best_property_query(self, query: str) -> bool:
        """Check if query is asking for 'best' properties.
        GENERIC - uses query_patterns from config!
        """
        best_keywords = self._settings.database.query_patterns.get("best_property", [])
        if not best_keywords:
            return False  # No patterns configured
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in best_keywords)

    def _prioritize_best_properties(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Prioritize properties based on boosting and verification status for 'best property' queries.
        GENERIC - uses boosting config from config!
        """
        boosting_config = self._settings.database.boosting
        
        # If boosting is disabled, just return chunks as is
        if not boosting_config or not boosting_config.enabled:
            return chunks
        
        fields = boosting_config.fields
        active_values = boosting_config.active_values
        
        def get_priority_score(chunk: RetrievedChunk) -> int:
            metadata = chunk.metadata
            score = 0
            
            # Check if all three boosting features are active
            premium_active = metadata.get(fields.premium) == active_values.premium
            carousel_active = metadata.get(fields.carousel) == active_values.carousel
            verification_active = metadata.get(fields.verification) == active_values.verification
            
            # Priority 1: All three boosting statuses active
            if premium_active and carousel_active and verification_active:
                score += 1000
            
            # Priority 2: Verified
            elif verification_active:
                score += 800
            
            # Priority 3: Carousel
            elif carousel_active:
                score += 600
            
            # Priority 4: Premium
            elif premium_active:
                score += 400
            
            # Add base score from vector similarity
            score += int(chunk.score * 100)
            
            return score
        
        # Sort by priority score (highest first)
        chunks.sort(key=get_priority_score, reverse=True)
        return chunks

    def _calculate_prioritized_boost(self, metadata: Dict[str, Any], query: str) -> float:
        """Enhanced boost for verified/premium properties with priority scoring.
        GENERIC - uses boosting config from config!
        """
        boosting_config = self._settings.database.boosting
        
        # If boosting is disabled, return 0
        if not boosting_config or not boosting_config.enabled:
            return 0.0
        
        boost = 0.0
        fields = boosting_config.fields
        active_values = boosting_config.active_values
        weights = boosting_config.boost_weights
        
        # Check each boosting feature
        premium_active = metadata.get(fields.premium) == active_values.premium
        carousel_active = metadata.get(fields.carousel) == active_values.carousel
        verification_active = metadata.get(fields.verification) == active_values.verification
        
        # Apply individual boosts
        if verification_active:
            boost += weights.verification
        
        if premium_active:
            boost += weights.premium
            
        if carousel_active:
            boost += weights.carousel
        
        # Additional boost for "best property" queries with all three active
        if self._is_best_property_query(query):
            if premium_active and carousel_active and verification_active:
                boost += weights.all_three
        
        return min(0.3, boost)  # Cap total boost at 0.3  
