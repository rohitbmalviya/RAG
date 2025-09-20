from __future__ import annotations
from typing import Dict, Tuple, Any, List
import re
import json
from .llm import LLMClient, is_best_property_query, is_average_price_query
from .config import get_settings

def extract_basic_filters(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Extract only the most basic filters using regex for performance.
    The comprehensive LLM extraction will handle everything else.
    """
    filters: Dict[str, Any] = {}
    q_lower = query.lower()

    # Only keep the most reliable regex patterns
    # Bedrooms (very reliable pattern)
    bed_match = re.search(r"(\d+)\s*bed(room)?s?", q_lower)
    if bed_match:
        filters["number_of_bedrooms"] = int(bed_match.group(1))

    # Bathrooms (very reliable pattern)
    bath_match = re.search(r"(\d+)\s*bath(room)?s?", q_lower)
    if bath_match:
        filters["number_of_bathrooms"] = int(bath_match.group(1))

    return filters

def _safe_convert_to_numeric(value: Any, target_type: str) -> Any:
    """Safely convert value to integer or float with fallback."""
    try:
        if target_type == "integer":
            return int(value)
        elif target_type == "float":
            return float(value)
    except Exception:
        pass
    return value

def _coerce_value_by_type(value: Any, field_type: str) -> Any:
    if value is None:
        return None
    t = (field_type or "").lower()
    if isinstance(value, dict):
        # range case for numeric fields
        coerced: Dict[str, Any] = {}
        for k in ("gte", "lte", "gt", "lt"):
            if k in value and value[k] is not None:
                if t in ("integer", "float"):
                    coerced[k] = _safe_convert_to_numeric(value[k], t)
                else:
                    coerced[k] = value[k]
        return coerced if coerced else None
    if t in ("integer", "float"):
        return _safe_convert_to_numeric(value, t)
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

    # First, apply basic regex extraction for simple patterns
    filters = extract_basic_filters(query, llm_client)

    schema_lines: List[str] = []
    for f in allowed_fields:
        ftype = field_types.get(f, "keyword")
        if ftype in ("integer", "float"):
            schema_lines.append(f"- {f} ({ftype}): number or range object {{'gte'|'lte'|'gt'|'lt'}}")
        elif ftype == "boolean":
            schema_lines.append(f"- {f} (boolean): true/false if explicitly implied")
        else:
            schema_lines.append(f"- {f} ({ftype}): string or list of strings")

    prompt = f"""Extract property search filters from the user query for a UAE property RAG system.
Return ONLY a JSON object, no prose. If nothing found, return {{}}.

AVAILABLE FIELDS:
{chr(10).join(schema_lines)}

=== COMPREHENSIVE FILTER EXTRACTION GUIDE ===

🏢 LOCATION FILTERS:
- EMIRATES: dubai, abu dhabi, sharjah, ajman, fujairah, ras al khaimah, umm al quwain
- HIERARCHY: emirate > city > community > subcommunity
- ABBREVIATIONS: JVC→jumeirah village circle, JBR→jumeirah beach residence, JLT→jumeirah lakes towers, DIFC→dubai international financial centre
- EXAMPLES: "Dubai Marina" → {{"emirate":"dubai", "community":"dubai marina"}}

🏠 PROPERTY TYPES:
- apartment/flat/condo → "apartment"
- villa/house → "villa"  
- studio → "studio"
- townhouse → "townhouse"
- duplex → "duplex"
- penthouse → "penthouse"
- office → "office"

💰 FINANCIAL FILTERS:
- RENT: "under 100k"→{{"rent_charge":{{"lte":100000}}}}, "150k-200k"→{{"rent_charge":{{"gte":150000,"lte":200000}}}}
- SECURITY DEPOSIT: "deposit 10k"→{{"security_deposit":{{"lte":10000}}}}
- MAINTENANCE: "maintenance 5k"→{{"maintenance_charge":{{"lte":5000}}}}
- CONVERSIONS: K=thousand, M=million

🛏️ PROPERTY SPECS:
- BEDROOMS: "2 bedroom"→{{"number_of_bedrooms":2}}
- BATHROOMS: "3 bathroom"→{{"number_of_bathrooms":3}}
- SIZE: "1500 sqft"→{{"property_size":1500}}, "built 2020"→{{"year_built":2020}}

🪑 FURNISHING & STATUS:
- FURNISHING: "furnished"/"semi-furnished"/"unfurnished"
- PROPERTY STATUS: "listed"/"active"/"draft"/"review"
- RENT TYPE: "lease"/"holiday home ready"/"management fees"
- MAINTENANCE: "owner"/"tenant"/"shared" (maintenance_covered_by)

⭐ VERIFICATION & BOOSTING:
- VERIFIED: "verified"→{{"bnb_verification_status":"verified"}}
- PREMIUM: "premium"→{{"premiumBoostingStatus":"Active"}}
- PRIME: "prime"→{{"carouselBoostingStatus":"Active"}}

🏊 AMENITIES (Boolean - set to true if mentioned):
- GYM: gym, fitness → gym_fitness_center
- POOL: pool, swimming → swimming_pool  
- PARKING: parking → parking
- BALCONY: balcony, terrace → balcony_terrace
- BEACH: beach access → beach_access
- ELEVATOR: elevator, lift → elevators
- SECURITY: security → security_available
- CONCIERGE: concierge → concierge_available
- MAID: maid room → maids_room
- LAUNDRY: laundry → laundry_room
- STORAGE: storage → storage_room
- BBQ: bbq → bbq_area
- PET: pet-friendly → pet_friendly
- AC: central ac, air conditioning → central_ac_heating
- SMART: smart home → smart_home_features
- WASTE: waste disposal → waste_disposal_system
- POWER: power backup, generator → power_backup
- MOSQUE: mosque nearby → mosque_nearby
- JOGGING: jogging, cycling tracks → jogging_cycling_tracks
- CHILLER: chiller included → chiller_included
- SUBLEASE: sublease allowed → sublease_allowed
- CHILDREN: kids play area → childrens_play_area

📅 DATE FILTERS (format as YYYY-MM-DD):
- AVAILABLE: "available from Jan 2025"→{{"available_from":"2025-01-01"}}
- LEASE START: "lease starts March"→{{"lease_start_date":"2025-03-01"}}
- LEASE END: "lease ends Dec 2025"→{{"lease_end_date":"2025-12-31"}}

🏗️ DEVELOPER & DETAILS:
- DEVELOPER: "Emaar", "Nakheel", "Damac" → developer_name
- LEASE DURATION: "1 year", "6 months", "2 years" → lease_duration
- FLOOR: "5th floor", "ground floor" → floor_level

EXTRACTION RULES:
1. Use lowercase for all string values
2. Only extract explicitly mentioned filters
3. For ranges, use gte/lte: {{"field":{{"gte":min,"lte":max}}}}
4. Boolean amenities: set to true only if clearly mentioned
5. Dates: convert to YYYY-MM-DD format
6. Don't invent values - only extract what's clearly stated

User query: {query}

Output JSON:"""

    raw = llm_client.generate(prompt).strip()
    
    def _safe_json_parse(text: str) -> Dict[str, Any]:
        """Safely parse JSON with multiple fallback strategies."""
        try:
            return json.loads(text)
        except Exception:
            pass
        return {}

    def _json_from_text(text: str) -> Dict[str, Any]:
        # Try direct JSON parsing first
        result = _safe_json_parse(text)
        if result:
            return result
        
        # Strip common markdown code fences
        if text.startswith("```") and text.endswith("```"):
            inner = text.strip().strip("`")
            # handle ```json ... ```
            lines = inner.split("\n", 1)
            if len(lines) == 2 and lines[0].strip().lower().startswith("json"):
                result = _safe_json_parse(lines[1])
                if result:
                    return result
        
        # Fallback: best-effort find first JSON object
        import re as _re
        m = _re.search(r"\{[\s\S]*\}", text)
        if m:
            result = _safe_json_parse(m.group(0))
            if result:
                return result
        
        return {}

    llm_data = _json_from_text(raw)

    # Merge hybrid filters with LLM results (LLM takes precedence for conflicts)
    result = filters.copy()
    
    def _normalize_string_value(value: Any, field_type: str) -> Any:
        """Normalize string values based on field type."""
        if not isinstance(value, (str, list)):
            return value
        
        if isinstance(value, str):
            value = value.strip()
            if field_type in ("keyword", "text"):
                return value.lower()
            return value
        elif isinstance(value, list):
            if field_type in ("keyword", "text"):
                return [str(item).lower() for item in value]
            return value
        return value

    # Process LLM results and merge
    for key, value in llm_data.items():
        if key not in allowed_fields:
            continue
        ftype = field_types.get(key, "")
        normalized_value = _normalize_string_value(value, ftype)
        coerced = _coerce_value_by_type(normalized_value, ftype)
        if coerced is not None:
            result[key] = coerced
    
    # Filter to only allowed fields
    result = {k: v for k, v in result.items() if k in allowed_fields}
    
    return result

def preprocess_query(query: str, llm_client: LLMClient) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query using LLM-only filter extraction restricted to configured fields.
    """
    filters = extract_filters_with_llm(query, llm_client)
    
    def _add_best_property_boosts(filters: Dict[str, Any]) -> None:
        """Add verification and boosting filters for best property queries."""
        if "bnb_verification_status" not in filters:
            filters["bnb_verification_status"] = "verified"
        if "premiumBoostingStatus" not in filters:
            filters["premiumBoostingStatus"] = "Active"

    # Add special handling for "best property" queries
    if is_best_property_query(query):
        _add_best_property_boosts(filters)
    
    print("filters", filters)
    return query, filters
