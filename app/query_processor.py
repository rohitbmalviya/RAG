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

⭐ VERIFICATION & BOOSTING (CRITICAL - Extract these based on user intent):
- VERIFIED PROPERTY: "verified property", "verified listings" → {{"bnb_verification_status":"verified"}}
- PREMIUM PROPERTY: "premium property", "premium listings", "premium" → {{"premiumBoostingStatus":"Active"}}
- PRIME PROPERTY: "prime property", "prime listings", "prime" → {{"carouselBoostingStatus":"Active"}}
- BEST PROPERTY: "best property", "best listings", "top property", "recommended" → {{"bnb_verification_status":"verified", "premiumBoostingStatus":"Active"}}

IMPORTANT: 
- When user says "prime" → use carouselBoostingStatus: "Active"
- When user says "premium" → use premiumBoostingStatus: "Active"  
- When user says "verified" → use bnb_verification_status: "verified"
- When user says "best" → use both verified + premium boosting

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
1. Use lowercase for all string values EXCEPT boosting status fields (use "Active" with capital A)
2. Only extract explicitly mentioned filters
3. For ranges, use gte/lte: {{"field":{{"gte":min,"lte":max}}}}
4. Boolean amenities: set to true only if clearly mentioned
5. Dates: convert to YYYY-MM-DD format
6. Don't invent values - only extract what's clearly stated
7. CRITICAL: Follow the exact boosting status mappings above - "prime"→carouselBoostingStatus, "premium"→premiumBoostingStatus, "verified"→bnb_verification_status, "best"→both verified+premium

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

def extract_filters_with_llm_context_aware(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Extract filters using LLM with conversation context awareness.
    This function uses dynamic configuration and scales to any property schema.
    """
    settings = get_settings()
    allowed_fields: List[str] = list(settings.retrieval.filter_fields or [])
    field_types: Dict[str, str] = dict(settings.database.field_types or {})

    # Get conversation context
    conversation_context = _get_conversation_context(llm_client)
    user_preferences = _get_user_preferences(llm_client)
    
    # First, apply basic regex extraction for simple patterns
    filters = extract_basic_filters(query, llm_client)

    # Get dynamic field mappings from configuration
    field_mappings = _get_dynamic_field_mappings(settings)
    
    # Build dynamic prompt based on actual schema
    prompt = _build_dynamic_extraction_prompt(conversation_context, user_preferences, field_mappings, query)

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

def _get_dynamic_field_mappings(settings) -> Dict[str, str]:
    """
    Get field mappings dynamically from configuration.
    This makes the system scalable to any property schema.
    """
    field_types = settings.database.field_types or {}
    filter_fields = settings.retrieval.filter_fields or []
    
    # Build dynamic mappings based on configuration
    mappings = {}
    
    for field in filter_fields:
        field_type = field_types.get(field, "keyword")
        mappings[field] = field_type
    
    return mappings

def _build_dynamic_extraction_prompt(conversation_context: str, user_preferences: Dict[str, Any], 
                                   field_mappings: Dict[str, str], query: str) -> str:
    """
    Build a dynamic extraction prompt based on the actual database schema.
    This approach scales to any property field configuration.
    """
    context_section = _build_context_section(conversation_context, user_preferences)
    
    # Build field descriptions dynamically
    field_descriptions = []
    for field, field_type in field_mappings.items():
        if field_type in ("integer", "float"):
            field_descriptions.append(f"- {field} ({field_type}): number or range object {{'gte'|'lte'|'gt'|'lt'}}")
        elif field_type == "boolean":
            field_descriptions.append(f"- {field} (boolean): true/false if explicitly implied")
        else:
            field_descriptions.append(f"- {field} ({field_type}): string or list of strings")
    
    prompt = f"""Extract property search filters from the user query for a UAE property RAG system.
Return ONLY a JSON object, no prose. If nothing found, return {{}}.

{context_section}

AVAILABLE FIELDS (dynamically configured):
{chr(10).join(field_descriptions)}

=== DYNAMIC EXTRACTION GUIDE ===

🏢 LOCATION FILTERS:
- Extract any UAE emirate, city, community, or subcommunity mentioned
- Examples: "Dubai Marina" → {{"emirate":"dubai", "community":"dubai marina"}}

💰 FINANCIAL FILTERS:
- Extract rent_charge, security_deposit, maintenance_charge as range objects
- Examples: "under 100k" → {{"rent_charge":{{"lte":100000}}}}

🛏️ PROPERTY SPECS:
- Extract number_of_bedrooms, number_of_bathrooms, property_size, year_built
- Examples: "2 bedroom" → {{"number_of_bedrooms":2}}

🪑 FURNISHING & STATUS:
- Extract furnishing_status, property_status, rent_type, maintenance_covered_by
- Examples: "furnished" → {{"furnishing_status":"furnished"}}

⭐ VERIFICATION & BOOSTING (CRITICAL - Extract these based on user intent):
- Extract bnb_verification_status, premiumBoostingStatus, carouselBoostingStatus
- Examples: 
  - "verified property", "verified listings", "verified" → {{"bnb_verification_status":"verified"}}
  - "premium property", "premium listings", "premium" → {{"premiumBoostingStatus":"Active"}}
  - "prime property", "prime listings", "prime" → {{"carouselBoostingStatus":"Active"}}
  - "best property", "best listings", "top property", "recommended" → {{"bnb_verification_status":"verified", "premiumBoostingStatus":"Active"}}

CRITICAL RULES:
- When user mentions "prime" → ALWAYS use carouselBoostingStatus: "Active"
- When user mentions "premium" → ALWAYS use premiumBoostingStatus: "Active"  
- When user mentions "verified" → ALWAYS use bnb_verification_status: "verified"
- When user mentions "best", "top", "recommended" → ALWAYS use both verified + premium boosting

🏊 AMENITIES (Boolean fields):
- Extract any boolean amenity fields mentioned
- Examples: "pool" → {{"swimming_pool":true}}

📅 DATE FILTERS:
- Extract date fields as range objects
- Examples: "available from Jan 2025" → {{"available_from":{{"gte":"2025-01-01"}}}}

User Query: {query}

Output JSON:"""
    
    return prompt

def preprocess_query(query: str, llm_client: LLMClient) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query using LLM-only filter extraction restricted to configured fields.
    The LLM handles all logic including boosting status mappings and case normalization.
    """
    # Extract filters with conversation context - LLM handles everything
    filters = extract_filters_with_llm_context_aware(query, llm_client)
    
    print("filters", filters)
    return query, filters
