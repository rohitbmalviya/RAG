from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests
import time
from .config import get_settings
from .utils import get_logger

logger = get_logger(__name__)

class RequirementGatherer:
    """Handles gathering and sending user requirements to admin endpoint"""
    
    def __init__(self):
        self.settings = get_settings()
        self.endpoint = getattr(self.settings, 'requirement_gathering', {}).get(
            'endpoint', 
            'http://localhost:5000/backend/api/v1/user/requirement'
        )
    
    def gather_requirements(
        self, 
        user_query: str, 
        preferences: Dict[str, Any], 
        conversation_summary: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Gather user requirements and send to endpoint"""
        
        requirement_data = {
            "user_query": user_query,
            "preferences": preferences,
            "conversation_summary": conversation_summary,
            "timestamp": time.time(),
            "session_id": session_id or "default"
        }
        
        try:
            response = requests.post(
                self.endpoint, 
                json=requirement_data, 
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully sent requirements for session {session_id}")
                return {
                    "status": "success",
                    "message": "Requirements saved successfully. Our team will work with agencies to find matching properties."
                }
            else:
                logger.error(f"Failed to send requirements: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": "Failed to save requirements. Please try again later."
                }
                
        except requests.exceptions.RequestException as exc:
            logger.error(f"Error sending requirements: {exc}")
            return {
                "status": "error",
                "message": "Failed to save requirements. Please try again later."
            }
    
    def suggest_alternatives(
        self, 
        user_preferences: Dict[str, Any], 
        available_locations: List[str]
    ) -> List[Dict[str, Any]]:
        """Suggest alternative search criteria when no matches found"""
        
        alternatives = []
        
        # Location alternatives
        if "emirate" in user_preferences:
            current_emirate = user_preferences["emirate"]
            if current_emirate == "sharjah":
                alternatives.append({
                    "type": "location",
                    "suggestion": "Try Dubai (nearby, more options available)",
                    "filters": {"emirate": "dubai"}
                })
            elif current_emirate == "dubai":
                alternatives.append({
                    "type": "location", 
                    "suggestion": "Try Abu Dhabi (similar lifestyle options)",
                    "filters": {"emirate": "abu dhabi"}
                })
        
        # Budget alternatives
        if "rent_charge" in user_preferences:
            current_budget = user_preferences["rent_charge"]
            if isinstance(current_budget, dict) and "lte" in current_budget:
                budget = current_budget["lte"]
                alternatives.append({
                    "type": "budget",
                    "suggestion": f"Try budget up to AED {int(budget * 1.2):,} (20% higher)",
                    "filters": {"rent_charge": {"lte": int(budget * 1.2)}}
                })
                alternatives.append({
                    "type": "budget",
                    "suggestion": f"Try budget up to AED {int(budget * 0.8):,} (20% lower)",
                    "filters": {"rent_charge": {"lte": int(budget * 0.8)}}
                })
        
        # Bedroom alternatives
        if "number_of_bedrooms" in user_preferences:
            bedrooms = user_preferences["number_of_bedrooms"]
            if bedrooms > 1:
                alternatives.append({
                    "type": "bedrooms",
                    "suggestion": f"Try {bedrooms - 1} bedroom properties",
                    "filters": {"number_of_bedrooms": bedrooms - 1}
                })
            alternatives.append({
                "type": "bedrooms",
                "suggestion": f"Try {bedrooms + 1} bedroom properties",
                "filters": {"number_of_bedrooms": bedrooms + 1}
            })
        
        # Furnishing alternatives
        if "furnishing_status" in user_preferences:
            current_furnishing = user_preferences["furnishing_status"]
            if current_furnishing == "furnished":
                alternatives.append({
                    "type": "furnishing",
                    "suggestion": "Try semi-furnished properties (more options available)",
                    "filters": {"furnishing_status": "semi-furnished"}
                })
            elif current_furnishing == "unfurnished":
                alternatives.append({
                    "type": "furnishing",
                    "suggestion": "Try furnished properties",
                    "filters": {"furnishing_status": "furnished"}
                })
        
        return alternatives[:3]  # Return top 3 alternatives
