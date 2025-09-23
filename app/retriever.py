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

    def _apply_prioritized_retrieval_strategy(self, filters: Optional[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
        """Apply prioritized context retrieval strategy based on user requirements."""
        if not filters:
            return filters
        
        # Priority order for context retrieval (as specified in requirements):
        # 1. Location (emirate, city, community, subcommunity, nearby_landmarks, public_transport_type)
        # 2. Budget (rent_charge, security_deposit, maintenance_charge)
        # 3. Property Type (property_type)
        # 4. Number of Bedrooms/Bathrooms (number_of_bedrooms, number_of_bathrooms)
        # 5. Furnishing Status (furnishing_status)
        # 6. Key Amenities (swimming_pool, pet_friendly, gym_fitness_center, parking, balcony_terrace)
        # 7. Lease Duration (lease_duration, rent_type)
        # 8. Availability Date (available_from)
        
        prioritized_filters = {}
        
        # 1. Location filters (highest priority)
        location_fields = ["emirate", "city", "community", "subcommunity", "nearby_landmarks", "public_transport_type"]
        for field in location_fields:
            if field in filters:
                prioritized_filters[field] = filters[field]
        
        # 2. Budget filters (high priority)
        budget_fields = ["rent_charge", "security_deposit", "maintenance_charge"]
        for field in budget_fields:
            if field in filters:
                prioritized_filters[field] = filters[field]
        
        # 3. Property type (high priority)
        if "property_type_name" in filters:
            prioritized_filters["property_type_name"] = filters["property_type_name"]
        
        # 4. Bedrooms/Bathrooms (medium-high priority)
        bedroom_bathroom_fields = ["number_of_bedrooms", "number_of_bathrooms"]
        for field in bedroom_bathroom_fields:
            if field in filters:
                prioritized_filters[field] = filters[field]
        
        # 5. Furnishing status (medium priority)
        if "furnishing_status" in filters:
            prioritized_filters["furnishing_status"] = filters["furnishing_status"]
        
        # 6. Key amenities (medium priority)
        amenity_fields = ["swimming_pool", "pet_friendly", "gym_fitness_center", "parking", "balcony_terrace"]
        for field in amenity_fields:
            if field in filters:
                prioritized_filters[field] = filters[field]
        
        # 7. Lease duration (medium-low priority)
        lease_fields = ["lease_duration", "rent_type_name"]
        for field in lease_fields:
            if field in filters:
                prioritized_filters[field] = filters[field]
        
        # 8. Availability date (low priority)
        if "available_from" in filters:
            prioritized_filters["available_from"] = filters["available_from"]
        
        # Add any remaining filters not covered above
        for key, value in filters.items():
            if key not in prioritized_filters:
                prioritized_filters[key] = value
        
        return prioritized_filters

    def _is_best_property_query(self, query: str) -> bool:
        """Check if query is asking for 'best' properties."""
        query_lower = query.lower()
        best_keywords = ["best", "top", "premium", "featured", "recommended", "highest quality"]
        return any(keyword in query_lower for keyword in best_keywords)

    def _prioritize_best_properties(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Prioritize properties based on boosting and verification status for 'best property' queries."""
        def get_priority_score(chunk: RetrievedChunk) -> int:
            metadata = chunk.metadata
            score = 0
            
            # Priority 1: All three boosting statuses active
            if (metadata.get("premiumBoostingStatus") == "Active" and 
                metadata.get("carouselBoostingStatus") == "Active" and 
                metadata.get("bnb_verification_status") == "verified"):
                score += 1000
            
            # Priority 2: Verified properties
            elif metadata.get("bnb_verification_status") == "verified":
                score += 800
            
            # Priority 3: Carousel boosting (Prime)
            elif metadata.get("carouselBoostingStatus") == "Active":
                score += 600
            
            # Priority 4: Premium boosting
            elif metadata.get("premiumBoostingStatus") == "Active":
                score += 400
            
            # Add base score from vector similarity
            score += int(chunk.score * 100)
            
            return score
        
        # Sort by priority score (highest first)
        chunks.sort(key=get_priority_score, reverse=True)
        return chunks

    def _calculate_prioritized_boost(self, metadata: Dict[str, Any], query: str) -> float:
        """Enhanced boost for verified/premium properties with priority scoring"""
        boost = 0.0
        
        # Enhanced boost for verified properties
        if metadata.get("bnb_verification_status") == "verified":
            boost += 0.15
        
        # Enhanced boost for premium properties
        if metadata.get("premiumBoostingStatus") == "Active":
            boost += 0.15
            
        # Enhanced boost for carousel properties
        if metadata.get("carouselBoostingStatus") == "Active":
            boost += 0.1
        
        # Additional boost for "best property" queries
        if self._is_best_property_query(query):
            if (metadata.get("premiumBoostingStatus") == "Active" and 
                metadata.get("carouselBoostingStatus") == "Active" and 
                metadata.get("bnb_verification_status") == "verified"):
                boost += 0.2  # Maximum boost for all three
        
        return min(0.3, boost)  # Cap total boost at 0.3  
