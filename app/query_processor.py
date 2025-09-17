from __future__ import annotations
from typing import Dict, Tuple, Any
import re
import json
from .models import RetrievedChunk
from .llm import LLMClient

# Controlled vocabularies
PROPERTY_TYPES = {
    "apartment": "Apartment",
    "villa": "Villa",
    "studio": "Studio",
    "duplex": "Duplex",
    "penthouse": "Penthouse",
    "office": "Office",
}

BOOSTING_KEYWORDS = {
    "verified": ("bnb_verification_status", "verified"),
    "premium": ("premiumBoostingStatus", "active"),
    "prime": ("carouselBoostingStatus", "active"),
}


def extract_rent_with_llm(query: str, llm_client: LLMClient) -> int | None:
    """
    Use LLM to extract the maximum rent value from a query.
    Converts shorthand like '10K', '2M', 'AED 15k' into integer (AED).
    """
    prompt = (
        "Extract the maximum rent value from this UAE property search query. "
        "Return only a number in AED. Convert 'K' to thousand, 'M' to million. "
        "If no rent specified, return None.\n\n"
        f"Query: {query}"
    )
    raw = llm_client.generate(prompt).strip()
    try:
        rent_value = int(raw.replace(",", "").replace(" ", ""))
        return rent_value
    except Exception:
        return None


def extract_location_with_llm(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Use LLM to extract structured location fields (emirate, city, district, community, subcommunity).
    """
    prompt = (
        "Extract location details from this UAE property search query.\n"
        "Return a JSON object with any of these fields if present:\n"
        "emirate, city, district, community, subcommunity.\n"
        "If not found, return an empty JSON.\n\n"
        f"Query: {query}"
    )
    raw = llm_client.generate(prompt).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v}
    except Exception:
        pass
    return {}


def extract_filters(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Extract structured filters from query using regex and LLM for rent.
    """
    filters: Dict[str, Any] = {}
    q_lower = query.lower()

    # Property type
    for key, value in PROPERTY_TYPES.items():
        if key in q_lower:
            filters["property_type_id"] = value

    # Bedrooms
    bed_match = re.search(r"(\d+)\s*bed(room)?s?", q_lower)
    if bed_match:
        filters["number_of_bedrooms"] = int(bed_match.group(1))

    # Bathrooms
    bath_match = re.search(r"(\d+)\s*bath(room)?s?", q_lower)
    if bath_match:
        filters["number_of_bathrooms"] = int(bath_match.group(1))

    # Property size
    size_match = re.search(r"(\d+)\s*(sqft|sqm)", q_lower)
    if size_match:
        filters["property_size"] = int(size_match.group(1))

    # Furnishing status
    if any(f in q_lower for f in ["semi-furnished", "semi furnished", "semifurnished"]):
        filters["furnishing_status"] = "semi-furnished"
    elif "unfurnished" in q_lower:
        filters["furnishing_status"] = "unfurnished"
    elif "furnished" in q_lower:
        filters["furnishing_status"] = "furnished"

    # Rent type
    if "holiday home" in q_lower:
        filters["rent_type_id"] = "Leasing - Holiday Home Ready"
    elif "management fee" in q_lower or "management fees" in q_lower:
        filters["rent_type_id"] = "Management Fees"
    elif "lease" in q_lower or "leasing" in q_lower:
        filters["rent_type_id"] = "Lease"

    # Boosting flags
    for keyword, (field, value) in BOOSTING_KEYWORDS.items():
        if keyword in q_lower:
            filters[field] = value

    # Rent charge via LLM (handles 10K, 2M, AED 5k)
    rent_value = extract_rent_with_llm(query, llm_client)
    if rent_value:
        filters["rent_charge"] = {"lte": rent_value}

    return filters


def preprocess_query(query: str, llm_client: LLMClient) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query: extract filters and location using regex + LLM.
    """
    filters = extract_filters(query, llm_client)

    # Add location via LLM
    location_filters = extract_location_with_llm(query, llm_client)
    filters.update(location_filters)

    return query, filters
