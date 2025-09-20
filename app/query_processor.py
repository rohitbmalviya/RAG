from __future__ import annotations
from typing import Dict, Tuple, Any, List
import re
import json
from .llm import LLMClient
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

    llm_data = _json_from_text(raw)

    # Merge hybrid filters with LLM results (LLM takes precedence for conflicts)
    result = filters.copy()
    
    # Process LLM results and merge
    for key, value in llm_data.items():
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
        if coerced is not None:
            result[key] = coerced
    
    # Filter to only allowed fields
    result = {k: v for k, v in result.items() if k in allowed_fields}
    
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
