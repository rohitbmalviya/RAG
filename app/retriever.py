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

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if top_k is None and (self._config.top_k in (None, 0)):
            self._logger.warning("Retrieval top_k not provided and config.top_k is missing/zero; defaulting to 5")
        k = top_k if top_k is not None else (self._config.top_k or 5)
        if self._config.num_candidates_multiplier in (None, 0):
            self._logger.warning("Retrieval num_candidates_multiplier is missing/zero; using 1x top_k")
        num_candidates = k * max(1, self._config.num_candidates_multiplier or 1)
        query_vec = self._embedder.embed([query], task_type="retrieval_query")[0]
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
            self._logger.error("Vector store search failed: %s", exc)
            raise
        chunks: List[RetrievedChunk] = []
        raw_scores: List[float] = [float(h.get("_score", 0.0)) for h in hits]
        max_score = max(raw_scores) if raw_scores else 1.0
        min_score = min(raw_scores) if raw_scores else 0.0
        denom = (max_score - min_score) if (max_score - min_score) > 0 else 1.0
        
        for hit in hits:
            raw = float(hit.get("_score", 0.0))
            base_score = (raw - min_score) / denom
            
            # Apply boosting for premium/verified properties
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            
            # Calculate priority score based on verification and boosting status
            priority_score = self._calculate_priority_score(metadata)
            
            # Calculate attribute priority score based on Step 7 requirements
            attribute_priority = self._calculate_attribute_priority_score(metadata, validated_filters)
            
            # Combine base similarity score with priority score and attribute priority
            final_score = min(1.0, base_score + priority_score + attribute_priority)
            
            chunks.append(
                RetrievedChunk(score=final_score, text=source.get("text", ""), metadata=metadata)
            )
        
        # Sort by final score (highest first)
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        return chunks

    def _calculate_priority_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate priority score based on verification and boosting status"""
        priority_score = 0.0
        
        # Priority 1: All three conditions met (premium + prime + verified)
        if (metadata.get("premiumBoostingStatus") == "Active" and 
            metadata.get("carouselBoostingStatus") == "Active" and 
            metadata.get("bnb_verification_status") == "verified"):
            priority_score = 0.3
        
        # Priority 2: Verified only
        elif metadata.get("bnb_verification_status") == "verified":
            priority_score = 0.2
        
        # Priority 3: Prime boosting only
        elif metadata.get("carouselBoostingStatus") == "Active":
            priority_score = 0.15
        
        # Priority 4: Premium boosting only
        elif metadata.get("premiumBoostingStatus") == "Active":
            priority_score = 0.1
        
        return priority_score

    def _calculate_attribute_priority_score(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> float:
        """
        Calculate attribute priority score based on Step 7 requirements.
        Prioritizes context retrieval based on user preferences and attribute importance.
        
        Priority Order (as specified in Step 7):
        1. Location (emirate, city, community) - Highest Priority
        2. Budget (rent_charge) - High Priority  
        3. Property Type - High Priority
        4. Bedrooms/Bathrooms - Medium-High Priority
        5. Furnishing Status - Medium Priority
        6. Key Amenities - Medium Priority
        7. Lease Duration - Medium-Low Priority
        8. Availability Date - Low Priority
        """
        if not filters:
            return 0.0
            
        attribute_score = 0.0
        
        # 1. Location Priority (Highest - 0.25 max)
        location_match = 0.0
        if "emirate" in filters and metadata.get("emirate") == filters["emirate"]:
            location_match += 0.15  # Emirate match
        if "city" in filters and metadata.get("city") == filters["city"]:
            location_match += 0.05  # City match
        if "community" in filters and metadata.get("community") == filters["community"]:
            location_match += 0.05  # Community match
        attribute_score += location_match
        
        # 2. Budget Priority (High - 0.15 max)
        if "rent_charge" in filters:
            filter_budget = filters["rent_charge"]
            property_rent = metadata.get("rent_charge")
            if property_rent and isinstance(filter_budget, dict):
                if "lte" in filter_budget:
                    max_budget = filter_budget["lte"]
                    if property_rent <= max_budget:
                        # Closer to budget gets higher score
                        budget_ratio = property_rent / max_budget
                        if budget_ratio <= 0.8:  # Well within budget
                            attribute_score += 0.15
                        elif budget_ratio <= 1.0:  # Within budget
                            attribute_score += 0.10
                        
        # 3. Property Type Priority (High - 0.12 max)
        if "property_type_id" in filters and metadata.get("property_type_id") == filters["property_type_id"]:
            attribute_score += 0.12
            
        # 4. Bedrooms/Bathrooms Priority (Medium-High - 0.10 max)
        if "number_of_bedrooms" in filters and metadata.get("number_of_bedrooms") == filters["number_of_bedrooms"]:
            attribute_score += 0.06
        if "number_of_bathrooms" in filters and metadata.get("number_of_bathrooms") == filters["number_of_bathrooms"]:
            attribute_score += 0.04
            
        # 5. Furnishing Status Priority (Medium - 0.08 max)
        if "furnishing_status" in filters and metadata.get("furnishing_status") == filters["furnishing_status"]:
            attribute_score += 0.08
            
        # 6. Key Amenities Priority (Medium - 0.06 max)
        amenity_matches = 0
        key_amenities = ["swimming_pool", "pet_friendly", "gym_fitness_center", "parking", "balcony_terrace"]
        for amenity in key_amenities:
            if amenity in filters and filters[amenity] and metadata.get(amenity):
                amenity_matches += 1
        if amenity_matches > 0:
            attribute_score += min(0.06, amenity_matches * 0.015)  # 0.015 per amenity match
            
        # 7. Lease Duration Priority (Medium-Low - 0.04 max)
        if "lease_duration" in filters and metadata.get("lease_duration") == filters["lease_duration"]:
            attribute_score += 0.04
            
        # 8. Availability Date Priority (Low - 0.02 max)
        if "available_from" in filters:
            filter_date = filters["available_from"]
            property_date = metadata.get("available_from")
            if property_date and filter_date:
                # Simple date comparison - exact match gets points
                if property_date <= filter_date:  # Available by requested date
                    attribute_score += 0.02
        
        return min(0.25, attribute_score)  # Cap at 0.25 to prevent over-boosting
