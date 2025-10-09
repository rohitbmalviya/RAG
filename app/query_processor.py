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

QUERY CATEGORY DETECTION (CRITICAL):
- "best property" or "best properties" or "top property" → {{"bnb_verification_status": "verified", "premiumBoostingStatus": "Active"}}
- "premium property" or "premium" → {{"premiumBoostingStatus": "Active"}}
- "prime property" or "prime" → {{"carouselBoostingStatus": "Active"}}
- "verified property" or "verified" → {{"bnb_verification_status": "verified"}}
- "average price" or "average rent" or "typical cost" → NO FILTERS (this is a calculation query)
- "hi", "hello", "good morning" → NO FILTERS (greeting query)
- "what is apartment" or "define villa" → NO FILTERS (general knowledge query)

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

PROPERTY TYPES (CRITICAL - Extract exact property type from user query):
- apartment/flat/condo → "apartment"
- villa/house → "villa"  
- studio → "studio"
- townhouse → "townhouse"
- duplex → "duplex"
- penthouse → "penthouse"
- office → "office"
- If user says "apartments" or "apartment" → MUST extract "property_type_name": "apartment"
- If user says "villas" or "villa" → MUST extract "property_type_name": "villa"
- If user says "studios" or "studio" → MUST extract "property_type_name": "studio"
- ALWAYS extract property type when explicitly mentioned in query

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

{amenity_rules}

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

CRITICAL PROPERTY TYPE EXAMPLES:
- "show me the property in dubai which are apartment" → {{"emirate": "dubai", "property_type_name": "apartment"}}
- "find apartments in dubai" → {{"emirate": "dubai", "property_type_name": "apartment"}}
- "show me villas in abu dhabi" → {{"emirate": "abu dhabi", "property_type_name": "villa"}}
- "studios in dubai marina" → {{"emirate": "dubai", "community": "dubai marina", "property_type_name": "studio"}}

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

CRITICAL: ACCUMULATE FILTERS FROM CONVERSATION:
- ALWAYS check conversation history for ALL previously mentioned filters
- ACCUMULATE filters from the entire conversation, don't just extract from current query
- If user previously mentioned location, property type, bedrooms, budget, etc., INCLUDE them all
- Example: User said "show me properties in Dubai" then "I want villa with 3 bedrooms"
  → Extract: {{"emirate": "dubai", "property_type_name": "villa", "number_of_bedrooms": 3}}
- Example: User said "properties in Dubai" then "apartment" then "2 bedrooms" then "under 150k"
  → Extract: {{"emirate": "dubai", "property_type_name": "apartment", "number_of_bedrooms": 2, "rent_charge": {{"lte": 150000}}}}
- Example: User said "show me properties in Dubai" then "show me apartments"
  → Extract: {{"emirate": "dubai", "property_type_name": "apartment"}}
- ALWAYS preserve ALL context from conversation history when user doesn't specify new values
- Only override filters if user explicitly mentions different values
- Look for ALL filter keywords: locations, property types, bedrooms, bathrooms, budget, amenities, etc.
- BUILD UPON previous filters, don't replace them unless explicitly changed
- If conversation shows "Dubai" was mentioned, ALWAYS include "emirate": "dubai" in your output

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

def _build_boolean_field_rules(settings) -> str:
    """Build boolean field extraction rules dynamically from configuration (domain-agnostic)."""
    feature_config = settings.database.boolean_fields or {}
    if not feature_config:
        return ""
    
    rules = ["BOOLEAN FEATURES (Boolean - set to true if mentioned):"]
    for field_name, display_label in feature_config.items():
        # Generate keywords from the display label
        keywords = display_label.lower()
        rules.append(f"- {display_label.upper()}: {keywords} → {field_name}")
    
    return "\n".join(rules)

def extract_filters_with_llm_context_aware(query: str, llm_client: LLMClient) -> Dict[str, Any]:
    """
    Extract filters using LLM-only approach with conversation context awareness.
    LLM handles all extraction logic including number conversions and format handling.
    This function uses dynamic configuration and scales to any property schema.
    """
    print(f"\n🔍 FILTER EXTRACTION DEBUG:")
    print(f"   Query: '{query}'")
    
    settings = get_settings()
    allowed_fields = list(settings.retrieval.filter_fields or [])
    field_types = dict(settings.database.field_types or {})
    
    # Get conversation context
    conversation_context = _get_conversation_context(llm_client)
    user_preferences = _get_user_preferences(llm_client)
    
    print(f"   Conversation context: '{conversation_context[:200]}...'")
    print(f"   User preferences: {user_preferences}")
    
    # Get existing filters from user preferences (stored from previous queries)
    existing_filters = user_preferences
    print(f"   Existing filters from user preferences: {existing_filters}")
    
    # Build dynamic prompt based on actual schema - LLM handles everything
    prompt = _build_dynamic_extraction_prompt(conversation_context, user_preferences, settings, query, existing_filters)
    print(f"   Prompt length: {len(prompt)} characters")
    print(f"   Prompt preview: {prompt[:300]}...")
    
    raw = llm_client.generate(prompt).strip()
    print(f"   LLM raw response: '{raw}'")
    
    llm_data = _json_from_text(raw)
    print(f"   Parsed JSON: {llm_data}")
    
    # CRITICAL FIX: Merge with existing filters - new filters override existing ones
    # But preserve existing filters that are not overridden
    merged_filters = {**existing_filters, **llm_data}
    print(f"   Merged filters: {merged_filters}")
    
    # DEBUG: Show what's being merged
    print(f"   DEBUG MERGE:")
    print(f"     Existing: {existing_filters}")
    print(f"     New: {llm_data}")
    print(f"     Final: {merged_filters}")
    
    # Process merged results with type coercion
    result = {}
    for key, value in merged_filters.items():
        if key not in allowed_fields:
            print(f"   Skipping invalid field: {key}")
            continue
        ftype = field_types.get(key, "")
        normalized_value = _normalize_string_value(value, ftype, key)
        coerced = _coerce_value_by_type(normalized_value, ftype)
        if coerced is not None:
            result[key] = coerced
            print(f"   Added filter: {key} = {coerced}")
    
    # Filter to only allowed fields
    result = {k: v for k, v in result.items() if k in allowed_fields}
    
    # Store the result in conversation for next query
    if hasattr(llm_client, 'conversation') and result:
        try:
            llm_client.conversation.update_preferences(result)
            print(f"   Stored filters in conversation for next query")
        except Exception as e:
            print(f"   Error storing filters: {e}")
    
    print(f"   Final filters: {result}")
    return result

def _get_conversation_context(llm_client: LLMClient) -> str:
    """Get recent conversation context for better filter extraction."""
    if not hasattr(llm_client, 'conversation'):
        return ""
    
    messages = llm_client.conversation.get_messages()
    if not messages:
        return ""
    
    # Get last 6 messages (3 user + 3 assistant) for better context
    recent_messages = messages[-6:]
    context_parts = []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            context_parts.append(f"User: {content}")
        elif msg["role"] == "assistant":
            # Extract key information from assistant responses
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            context_parts.append(f"Assistant: {content}")
    
    context = " | ".join(context_parts)
    print(f"   Conversation context extracted: {len(context)} characters")
    return context

def _get_user_preferences(llm_client: LLMClient) -> Dict[str, Any]:
    """Get user preferences from conversation history."""
    if not hasattr(llm_client, 'conversation'):
        return {}
    
    return llm_client.conversation.get_preferences()


def _build_context_section(conversation_context: str, user_preferences: Dict[str, Any], existing_filters: Dict[str, Any] = None) -> str:
    """Build context section for the LLM prompt."""
    context_sections = []
    
    if conversation_context:
        context_sections.append(f"CONVERSATION HISTORY:\n{conversation_context}\n\nCRITICAL: Extract ALL filters mentioned in this conversation history, not just the current query. Build upon previous filters. If user previously mentioned location, property type, bedrooms, budget, etc., INCLUDE them all in your extraction. For example, if the conversation shows 'Dubai' was mentioned, you MUST include 'emirate': 'dubai' in your output.")
    
    if existing_filters:
        filter_items = []
        for key, value in existing_filters.items():
            if isinstance(value, dict) and "lte" in value:
                filter_items.append(f"{key}: up to {value['lte']}")
            elif isinstance(value, dict) and "gte" in value:
                filter_items.append(f"{key}: at least {value['gte']}")
            else:
                filter_items.append(f"{key}: {value}")
        if filter_items:
            context_sections.append(f"EXISTING FILTERS FROM CONVERSATION: {', '.join(filter_items)}\n\nCRITICAL: These filters MUST be included in your output. Only override them if the current query explicitly changes them.")
    
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
                                   settings, query: str, existing_filters: Dict[str, Any] = None) -> str:
    """
    Build a dynamic extraction prompt based on the actual database schema.
    This approach scales to any property field configuration.
    """
    context_section = _build_context_section(conversation_context, user_preferences, existing_filters)
    field_descriptions = _build_field_descriptions(settings)
    boolean_rules = _build_boolean_field_rules(settings)
    
    # Build the complete prompt with context
    base_instructions = LLM_FILTER_EXTRACTION_INSTRUCTIONS.format(
        field_descriptions=field_descriptions,
        amenity_rules=boolean_rules,
        query=query
    )
    
    # Insert context section before the final instructions
    if context_section:
        # Find the position to insert context (before "User query:")
        insert_pos = base_instructions.find("User query:")
        if insert_pos != -1:
            final_prompt = base_instructions[:insert_pos] + context_section + base_instructions[insert_pos:]
            print(f"   Context section added: {len(context_section)} characters")
            return final_prompt
        else:
            print(f"   Warning: Could not find 'User query:' in prompt")
    
    print(f"   No context section added")
    return base_instructions

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

    print("🔍 FILTER EXTRACTION DEBUG:")
    print(f"   Query: '{query}'")
    print(f"   Extracted filters: {filters}")
    
    return query, filters
