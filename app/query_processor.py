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


# Filters supported by retrieval (must match config.yaml retrieval.filter_fields)
ALLOWED_FILTER_KEYS = {
    "emirate",
    "city",
    "district",
    "community",
    "subcommunity",
    "property_type_id",
    "rent_type_id",
    "furnishing_status",
    "number_of_bedrooms",
    "number_of_bathrooms",
    "rent_charge",
    "property_size",
    "bnb_verification_status",
    "premiumBoostingStatus",
    "carouselBoostingStatus",
}

NUMERIC_KEYS_FLOAT = {"rent_charge", "property_size"}
NUMERIC_KEYS_INT = {"number_of_bedrooms", "number_of_bathrooms"}


def _normalize_filter_value(key: str, value: Any) -> Any:
    """Coerce LLM outputs to expected ES filter formats (term/terms/range)."""
    if value is None:
        return None
    if key in NUMERIC_KEYS_FLOAT:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(",", "").strip())
            except Exception:
                return None
        if isinstance(value, dict):
            rng: Dict[str, Any] = {}
            for b in ("gte", "lte", "gt", "lt"):
                if b in value:
                    try:
                        rng[b] = float(str(value[b]).replace(",", "").strip())
                    except Exception:
                        pass
            return rng if rng else None
        return None
    if key in NUMERIC_KEYS_INT:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.replace(",", "").strip()))
            except Exception:
                return None
        if isinstance(value, dict):
            rng = {}
            for b in ("gte", "lte", "gt", "lt"):
                if b in value:
                    try:
                        rng[b] = int(float(str(value[b]).replace(",", "").strip()))
                    except Exception:
                        pass
            return rng if rng else None
        return None
    # Non-numeric fields: accept string or list of strings
    if isinstance(value, str):
        val = value.strip()
        return val if val else None
    if isinstance(value, list):
        vals = [str(v).strip() for v in value if str(v).strip()]
        return vals if vals else None
    return None


def _extract_filters_with_llm(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """Ask LLM to output filters as strict JSON, then validate/normalize."""
    schema_keys = ", ".join(sorted(list(ALLOWED_FILTER_KEYS)))
    prompt = (
        "From this UAE property search query, extract structured filters.\n"
        "Return ONLY a compact JSON object with any of these keys if present: \n"
        f"{schema_keys}\n"
        "Value rules:\n"
        "- For numeric fields (number_of_bedrooms, number_of_bathrooms), return an integer or a range object {\"gte\": int, \"lte\": int}.\n"
        "- For numeric fields (property_size, rent_charge), return a number or a range object {\"gte\": float, \"lte\": float}.\n"
        "- For phrases like 'around ~5000 sqft', use a ±10% range for property_size.\n"
        "- For furnishing, use furnished | unfurnished | semi-furnished.\n"
        "- Omit unknown fields (do not include null or empty).\n\n"
        f"Query: {query}"
    )
    raw = llm_client.generate(prompt).strip()
    data = _parse_first_json_object(raw)
    if not data:
        return {}
    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        if k in ALLOWED_FILTER_KEYS:
            norm = _normalize_filter_value(k, v)
            if norm is not None and not (isinstance(norm, dict) and not norm):
                cleaned[k] = norm
    return cleaned


def _parse_first_json_object(raw: str) -> Dict[str, Any]:
    """Best-effort to extract the first JSON object from an LLM response."""
    text = raw.strip()
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    # Find first {...}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    Use LLM to extract structured UAE location fields with a strict JSON schema.
    """
    prompt = (
        "Extract UAE location fields from the user's property search query.\n"
        "Respond ONLY with a compact JSON object matching this schema: \n"
        "{""emirate"": string|omit, ""city"": string|omit, ""district"": string|omit, ""community"": string|omit, ""subcommunity"": string|omit}\n"
        "Rules:\n"
        "- If a field is unknown, omit it (do not include null or empty).\n"
        "- Normalize spellings (e.g., 'abu dhabhi' -> 'Abu Dhabi').\n"
        "- Only include UAE places. If outside UAE, return {}.\n\n"
        f"Query: {query}"
    )
    raw = llm_client.generate(prompt).strip()
    data = _parse_first_json_object(raw)
    if not data:
        return {}
    # Keep only expected keys with non-empty values
    allowed_keys = {"emirate", "city", "district", "community", "subcommunity"}
    cleaned: Dict[str, Any] = {k: v for k, v in data.items() if k in allowed_keys and isinstance(v, str) and v.strip()}
    return cleaned


def extract_filters(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Extract structured filters using the LLM (robust JSON), with minimal keyword fallback for boosting flags.
    """
    filters = _extract_filters_with_llm(query, llm_client)
    q_lower = query.lower()
    # Minimal keyword fallback for boosting flags (simple and deterministic)
    for keyword, (field, value) in BOOSTING_KEYWORDS.items():
        if keyword in q_lower and field not in filters:
            filters[field] = value
    return filters


def preprocess_query(query: str, llm_client: LLMClient) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query: extract filters and location using regex + LLM.
    """
    filters = extract_filters(query, llm_client)

    # Add LLM-derived location (no deterministic hardcoding)
    location_filters = extract_location_with_llm(query, llm_client)
    filters.update(location_filters)

    return query, filters
