from __future__ import annotations
from typing import Dict, Tuple, Any, List
import re
import json
from .models import RetrievedChunk
from .llm import LLMClient
from .config import get_settings

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


def _coerce_value_by_type(value: Any, field_type: str) -> Any:
    if value is None:
        return None
    t = (field_type or "").lower()
    if isinstance(value, dict):
        # range case for numeric fields
        coerced: Dict[str, Any] = {}
        for k in ("gte", "lte", "gt", "lt"):
            if k in value and value[k] is not None:
                try:
                    if t == "integer":
                        coerced[k] = int(value[k])
                    elif t == "float":
                        coerced[k] = float(value[k])
                    else:
                        coerced[k] = value[k]
                except Exception:
                    coerced[k] = value[k]
        return coerced if coerced else None
    if t == "integer":
        try:
            return int(value)
        except Exception:
            return value
    if t == "float":
        try:
            return float(value)
        except Exception:
            return value
    if t == "boolean":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        return s in {"1", "true", "yes", "on"}
    # keyword/text remain as-is (strings or lists)
    return value


def extract_filters_with_llm(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Use LLM to extract filters strictly limited to retrieval.filter_fields and
    coerced to their types from database.field_types. Returns a dict suitable
    for the retriever (supports exact, list, or range objects for numeric).
    """
    settings = get_settings()
    allowed_fields: List[str] = list(settings.retrieval.filter_fields or [])
    field_types: Dict[str, str] = dict(settings.database.field_types or {})

    schema_lines: List[str] = []
    for f in allowed_fields:
        ftype = field_types.get(f, "keyword")
        if ftype in ("integer", "float"):
            schema_lines.append(f"- {f} ({ftype}): number or range object {{'gte'|'lte'|'gt'|'lt'}}")
        elif ftype == "boolean":
            schema_lines.append(f"- {f} (boolean): true/false if explicitly implied")
        else:
            schema_lines.append(f"- {f} ({ftype}): string or list of strings")

    prompt = (
        "Extract property search filters from the user query for a UAE property RAG system.\n"
        "Return ONLY a JSON object, no prose. If nothing found, return {}.\n"
        "Include only these fields if explicitly implied (omit others):\n"
        + "\n".join(schema_lines)
        + "\n\nRules:\n"
          "- Use lowercase strings for categorical values.\n"
          "- For numbers with K/M (e.g., 120k, 1.2M), convert to AED integer/float.\n"
          "- For ceilings like 'under 120k', output rent_charge as {\"lte\": 120000}.\n"
          "- Map 'verified' to bnb_verification_status: 'verified'.\n"
          "- Map 'premium' to premiumBoostingStatus: 'Active'.\n"
          "- Map 'prime' to carouselBoostingStatus: 'Active'.\n"
          "- Booleans like 'pet-friendly', 'gym', 'beach access' -> true.\n"
          "- For 'best property' queries, prioritize verified and boosted properties.\n"
          "- For location queries, map to emirate, city, community, or subcommunity.\n"
          "- Do not invent values. If not clearly implied, omit the field.\n\n"
        f"User query: {query}\n\nOutput JSON:"
    )

    raw = llm_client.generate(prompt).strip()
    data: Dict[str, Any] = {}

    def _json_from_text(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        # Strip common markdown code fences
        if text.startswith("```") and text.endswith("```"):
            inner = text.strip().strip("`")
            # handle ```json ... ```
            lines = inner.split("\n", 1)
            if len(lines) == 2 and lines[0].strip().lower().startswith("json"):
                try:
                    return json.loads(lines[1])
                except Exception:
                    pass
        # Fallback: best-effort find first JSON object
        import re as _re
        m = _re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

    data = _json_from_text(raw)

    # Keep only allowed fields and coerce by type
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in allowed_fields:
            continue
        # Normalize categorical strings to lowercase
        v = value
        if isinstance(v, str):
            v = v.strip()
        ftype = field_types.get(key, "")
        if ftype in ("keyword", "text"):
            if isinstance(v, str):
                v = v.lower()
            elif isinstance(v, list):
                v = [str(item).lower() for item in v]
        coerced = _coerce_value_by_type(v, ftype)
        if coerced is None:
            continue
        result[key] = coerced
    # Fallback to regex extraction if LLM returned empty
    if not result:
        try:
            fallback = extract_filters(query, llm_client)
            if fallback:
                # Keep only allowed fields
                result = {k: v for k, v in fallback.items() if k in allowed_fields}
        except Exception:
            pass
    return result


def is_best_property_query(query: str) -> bool:
    """Check if the query is asking for 'best' properties"""
    query = query.lower().strip()
    best_keywords = ["best", "top", "premium", "featured", "recommended", "highest rated"]
    return any(keyword in query for keyword in best_keywords)

def is_average_price_query(query: str) -> bool:
    """Check if the query is asking for average prices"""
    query = query.lower().strip()
    price_keywords = ["average", "mean", "typical", "usual", "normal"]
    price_indicators = ["price", "rent", "cost", "rate"]
    return any(keyword in query for keyword in price_keywords) and any(indicator in query for indicator in price_indicators)

def preprocess_query(query: str, llm_client: LLMClient) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query using LLM-only filter extraction restricted to configured fields.
    """
    filters = extract_filters_with_llm(query, llm_client)
    
    # Add special handling for "best property" queries
    if is_best_property_query(query):
        # Prioritize verified and boosted properties
        if "bnb_verification_status" not in filters:
            filters["bnb_verification_status"] = "verified"
        if "premiumBoostingStatus" not in filters:
            filters["premiumBoostingStatus"] = "Active"
    
    print("filters", filters)
    return query, filters
