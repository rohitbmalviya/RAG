from __future__ import annotations
from typing import Dict, Tuple, Any, List
import json
from .llm import LLMClient
from .config import get_settings

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

# Constants for LLM instructions - centralized for maintainability
LLM_FILTER_EXTRACTION_INSTRUCTIONS = """Extract property search filters from the user query for a UAE property RAG system.
Return ONLY a JSON object, no prose. If nothing found, return {{}}.

AVAILABLE FIELDS:
{field_descriptions}

=== COMPREHENSIVE FILTER EXTRACTION GUIDE ===

CRITICAL IMPROVEMENTS FOR ACCURACY:

BOOSTING STATUS DETECTION (VERY IMPORTANT):
- "best property" or "best properties" → {{"bnb_verification_status": "verified", "premiumBoostingStatus": "Active"}}
- "top property" or "recommended" → {{"bnb_verification_status": "verified", "premiumBoostingStatus": "Active"}}  
- "premium property" or "premium" → {{"premiumBoostingStatus": "Active"}}
- "prime property" or "prime" → {{"carouselBoostingStatus": "Active"}}
- "verified property" or "verified" → {{"bnb_verification_status": "verified"}}

LOCATION EXTRACTION (ENHANCED):
- Always extract emirate AND community when both mentioned
- "Dubai Marina" → {{"emirate": "dubai", "community": "dubai marina"}}
- "JBR" or "Jumeirah Beach Residence" → {{"emirate": "dubai", "community": "jumeirah beach residence"}}
- "Downtown Dubai" → {{"emirate": "dubai", "community": "downtown dubai"}}
- "Business Bay" → {{"emirate": "dubai", "community": "business bay"}}
- "JLT" or "Jumeirah Lakes Towers" → {{"emirate": "dubai", "community": "jumeirah lakes towers"}}
- "JVC" or "Jumeirah Village Circle" → {{"emirate": "dubai", "community": "jumeirah village circle"}}
- "DIFC" or "Dubai International Financial Centre" → {{"emirate": "dubai", "community": "dubai international financial centre"}}

PROPERTY TYPES:
- apartment/flat/condo → "apartment"
- villa/house → "villa"  
- studio → "studio"
- townhouse → "townhouse"
- duplex → "duplex"
- penthouse → "penthouse"
- office → "office"

FINANCIAL FILTERS (CRITICAL - Handle ALL number formats intelligently):
- RENT AMOUNTS: Extract rent_charge with proper conversion and range logic
- NUMBER FORMATS TO HANDLE:
  * "100000" → 100000 (direct number)
  * "100k" → 100000 (k = thousand)
  * "100 K" → 100000 (space before K)
  * "100000 AED" → 100000 (with currency)
  * "100k AED" → 100000 (k + currency)
  * "1.5M" → 1500000 (M = million)
  * "1,500,000" → 1500000 (comma-separated)
- RANGE PATTERNS:
  * "under/below/max/up to X" → {{"rent_charge":{{"lte":X}}}}
  * "above/over/at least X" → {{"rent_charge":{{"gte":X}}}}
  * "between X and Y" → {{"rent_charge":{{"gte":X,"lte":Y}}}}
  * "X to Y" → {{"rent_charge":{{"gte":X,"lte":Y}}}}
  * "X-Y" → {{"rent_charge":{{"gte":X,"lte":Y}}}}
- EXAMPLES:
  * "under 800000 AED" → {{"rent_charge":{{"lte":800000}}}}
  * "below 100k" → {{"rent_charge":{{"lte":100000}}}}
  * "100 K AED" → {{"rent_charge":{{"lte":100000}}}}
  * "between 150k and 200k AED" → {{"rent_charge":{{"gte":150000,"lte":200000}}}}
  * "500k to 800k" → {{"rent_charge":{{"gte":500000,"lte":800000}}}}
- OTHER FINANCIAL FIELDS:
  * SECURITY DEPOSIT: "deposit 10k"→{{"security_deposit":{{"lte":10000}}}}
  * MAINTENANCE: "maintenance 5k"→{{"maintenance_charge":{{"lte":5000}}}}
- CONVERSION RULES: Always convert to final numbers (no K/M suffixes in output)

PROPERTY SPECS (Handle ALL number formats):
- BEDROOMS: "2 bedroom"/"2 bed"/"2BR"→{{"number_of_bedrooms":2}}
- BATHROOMS: "3 bathroom"/"3 bath"/"3BA"→{{"number_of_bathrooms":3}}
- SIZE: "1500 sqft"/"1500 sq ft"/"1500 square feet"→{{"property_size":1500}}
- YEAR BUILT: "built 2020"/"constructed 2018"/"2020 built"→{{"year_built":2020}}
- FLOOR: "5th floor"/"ground floor"/"level 3"→{{"floor_level":"5th floor"}}
- PLOT NUMBER: "plot 123"/"plot number 456"→{{"plot_number":123}}
- UNIT NUMBER: "unit 789"/"apartment 101"→{{"apartment_unit_number":"789"}}

FURNISHING & STATUS:
- FURNISHING: "furnished"/"semi-furnished"/"unfurnished"
- PROPERTY STATUS: "listed"/"active"/"draft"/"review"
- RENT TYPE: "lease"/"holiday home ready"/"management fees"
- MAINTENANCE: "owner"/"tenant"/"shared" (maintenance_covered_by)

AMENITIES (Boolean - set to true if mentioned):
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

DATE FILTERS (format as YYYY-MM-DD):
- AVAILABLE: "available from Jan 2025"→{{"available_from":"2025-01-01"}}
- LEASE START: "lease starts March"→{{"lease_start_date":"2025-03-01"}}
- LEASE END: "lease ends Dec 2025"→{{"lease_end_date":"2025-12-31"}}

DEVELOPER & DETAILS:
- DEVELOPER: "Emaar", "Nakheel", "Damac" → developer_name
- LEASE DURATION: "1 year", "6 months", "2 years", "12-month" → lease_duration
- FLOOR: "5th floor", "ground floor" → floor_level
- PLOT NUMBER: "plot 123" → plot_number
- UNIT NUMBER: "unit 456", "apartment 789" → apartment_unit_number

ENHANCED LOCATION EXTRACTION:
- NEARBY LANDMARKS: "near metro", "close to mall", "near Sheikh Zayed Road" → nearby_landmarks
- TRANSPORT: "metro access", "bus station nearby" → public_transport_type
- BEACH ACCESS: "beach front", "near beach" → beach_access

ENHANCED AMENITY EXTRACTION:
- PRIVATE POOL: "private pool", "own pool" → swimming_pool with private indicator
- SHARED POOL: "shared pool", "community pool" → swimming_pool with shared indicator
- PARKING DETAILS: "3 parking", "covered parking" → parking with details
- FURNISHING DETAILS: "fully furnished", "partially furnished" → furnishing_status

EXTRACTION RULES:
1. Use lowercase for all string values EXCEPT boosting status fields (use "Active" with capital A)
2. Only extract explicitly mentioned filters - don't invent values
3. For ranges, use gte/lte: {{"field":{{"gte":min,"lte":max}}}}
4. Boolean amenities: set to true only if clearly mentioned
5. Dates: convert to YYYY-MM-DD format
6. CRITICAL: Follow the exact boosting status mappings above - "prime"→carouselBoostingStatus, "premium"→premiumBoostingStatus, "verified"→bnb_verification_status, "best"→both verified+premium

EXACT VALUE MATCHING:
- Use exact enum values: "furnished", "semi-furnished", "unfurnished"
- Boosting status: "Active", "verified" (case-sensitive)
- Property status: "listed" (case-sensitive)

RANGE HANDLING (IMPROVED):
- "under 100k" → {{"rent_charge": {{"lte": 100000}}}}
- "above 50k" → {{"rent_charge": {{"gte": 50000}}}}  
- "between 80k and 120k" → {{"rent_charge": {{"gte": 80000, "lte": 120000}}}}

INTELLIGENT NUMBER CONVERSION (Handle ALL formats):
- Convert "100 K" → 100000 (space + K = thousand)
- Convert "100k" → 100000 (no space + k = thousand)  
- Convert "1.5M" → 1500000 (M = million)
- Convert "1,500,000" → 1500000 (remove commas)
- Convert "100000 AED" → 100000 (remove currency)
- Convert "100 K AED" → 100000 (space + K + currency)
- Always output final numbers (no K/M suffixes in JSON)
- Handle mixed formats: "150k to 200k AED" → {{"gte":150000,"lte":200000}}

CONTEXT-AWARE EXTRACTION:
- If user says "budget is 100k", extract as rent_charge upper limit
- If user says "minimum 50k", extract as rent_charge lower limit  
- If user says "around 100k", extract as range with ±10% tolerance
- Consider conversation context for implicit filters
- Use user preferences from previous messages when relevant

User query: {query}

Output JSON:"""

# Boosting status fields that preserve case
BOOSTING_FIELDS = {"bnb_verification_status", "premiumBoostingStatus", "carouselBoostingStatus"}

def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Safely parse JSON with multiple fallback strategies."""
    try:
        return json.loads(text)
    except Exception:
        pass
    return {}

def _json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text with multiple fallback strategies."""
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

def _normalize_string_value(value: Any, field_type: str, field_name: str = "") -> Any:
    """Normalize string values based on field type."""
    if not isinstance(value, (str, list)):
        return value
    
    if isinstance(value, str):
        value = value.strip()
        if field_type in ("keyword", "text") and field_name not in BOOSTING_FIELDS:
            return value.lower()
        return value
    elif isinstance(value, list):
        if field_type in ("keyword", "text") and field_name not in BOOSTING_FIELDS:
            return [str(item).lower() for item in value]
        return value
    return value

def _build_field_descriptions(settings) -> str:
    """Build field descriptions dynamically from configuration."""
    allowed_fields = list(settings.retrieval.filter_fields or [])
    field_types = dict(settings.database.field_types or {})
    schema_lines: List[str] = []
    for f in allowed_fields:
        ftype = field_types.get(f, "keyword")
        if ftype in ("integer", "float"):
            schema_lines.append(f"- {f} ({ftype}): number or range object {{'gte'|'lte'|'gt'|'lt'}}")
        elif ftype == "boolean":
            schema_lines.append(f"- {f} (boolean): true/false if explicitly implied")
        else:
            schema_lines.append(f"- {f} ({ftype}): string or list of strings")
    
    return "\n".join(schema_lines)

def extract_filters_with_llm_context_aware(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Extract filters using LLM-only approach with conversation context awareness.
    LLM handles all extraction logic including number conversions and format handling.
    This function uses dynamic configuration and scales to any property schema.
    """
    settings = get_settings()
    allowed_fields = list(settings.retrieval.filter_fields or [])
    field_types = dict(settings.database.field_types or {})
    
    # Get conversation context
    conversation_context = _get_conversation_context(llm_client)
    user_preferences = _get_user_preferences(llm_client)
    
    # Build dynamic prompt based on actual schema - LLM handles everything
    prompt = _build_dynamic_extraction_prompt(conversation_context, user_preferences, settings, query)
    raw = llm_client.generate(prompt).strip()
    llm_data = _json_from_text(raw)
    
    # Process LLM results with type coercion
    result = {}
    for key, value in llm_data.items():
        if key not in allowed_fields:
            continue
        ftype = field_types.get(key, "")
        normalized_value = _normalize_string_value(value, ftype, key)
        coerced = _coerce_value_by_type(normalized_value, ftype)
        if coerced is not None:
            result[key] = coerced
    
    # Filter to only allowed fields
    result = {k: v for k, v in result.items() if k in allowed_fields}
    
    return result

def _get_conversation_context(llm_client: LLMClient) -> str:
    """Get recent conversation context for better filter extraction."""
    if not hasattr(llm_client, 'conversation'):
        return ""
    
    messages = llm_client.conversation.get_messages()
    if not messages:
        return ""
    
    # Get last 4 messages (2 user + 2 assistant) for context
    recent_messages = messages[-4:]
    context_parts = []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            context_parts.append(f"User: {content}")
        elif msg["role"] == "assistant":
            # Extract key information from assistant responses
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            context_parts.append(f"Assistant: {content}")
    
    return " | ".join(context_parts)

def _get_user_preferences(llm_client: LLMClient) -> Dict[str, Any]:
    """Get user preferences from conversation history."""
    if not hasattr(llm_client, 'conversation'):
        return {}
    
    return llm_client.conversation.get_preferences()

def _build_context_section(conversation_context: str, user_preferences: Dict[str, Any]) -> str:
    """Build context section for the LLM prompt."""
    context_sections = []
    
    if conversation_context:
        context_sections.append(f"CONVERSATION HISTORY:\n{conversation_context}")
    
    if user_preferences:
        pref_items = []
        for key, value in user_preferences.items():
            if isinstance(value, dict) and "lte" in value:
                pref_items.append(f"{key}: up to {value['lte']}")
            elif isinstance(value, dict) and "gte" in value:
                pref_items.append(f"{key}: at least {value['gte']}")
            else:
                pref_items.append(f"{key}: {value}")
        if pref_items:
            context_sections.append(f"USER PREFERENCES: {', '.join(pref_items)}")
    
    if context_sections:
        return "\n\n".join(context_sections) + "\n\n"
    
    return ""

def _build_dynamic_extraction_prompt(conversation_context: str, user_preferences: Dict[str, Any], 
                                   settings, query: str) -> str:
    """
    Build a dynamic extraction prompt based on the actual database schema.
    This approach scales to any property field configuration.
    """
    context_section = _build_context_section(conversation_context, user_preferences)
    field_descriptions = _build_field_descriptions(settings)
    
    return LLM_FILTER_EXTRACTION_INSTRUCTIONS.format(
        field_descriptions=field_descriptions,
        context_section=context_section,
        query=query
    )

def preprocess_query(query: str, llm_client: LLMClient) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query using LLM-only filter extraction.
    LLM handles ALL extraction logic including:
    - Number format conversions (100k → 100000, 100 K → 100000, 1.5M → 1500000)
    - Range detection (under/below/above/between patterns)
    - Currency handling (AED amounts)
    - Context-aware extraction from conversation history
    - All filter types with superior flexibility vs regex patterns
    """
    # Extract filters with conversation context - LLM handles everything intelligently
    filters = extract_filters_with_llm_context_aware(query, llm_client)

    print("Extracted filters:", filters)
    
    return query, filters
