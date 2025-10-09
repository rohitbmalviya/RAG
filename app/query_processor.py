from __future__ import annotations
from typing import Dict, Tuple, Any, List
import json
from .llm import LLMClient
from .config import get_settings, logger
# Import instructions from centralized file
from .instructions import LLM_FILTER_EXTRACTION_INSTRUCTIONS

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

# Boosting status fields that preserve case (imported from instructions.py, rest of instructions removed - NO DUPLICATES!)
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
    logger.debug(f"\n FILTER EXTRACTION DEBUG:")
    logger.debug(f" Query: '{query}'")
    settings = get_settings()
    allowed_fields = list(settings.retrieval.filter_fields or [])
    field_types = dict(settings.database.field_types or {})
    
    # Get conversation context
    conversation_context = _get_conversation_context(llm_client)
    user_preferences = _get_user_preferences(llm_client)
    
    logger.debug(f" Conversation context: '{conversation_context[:200]}...'")
    logger.debug(f" User preferences: {user_preferences}")
    # Get existing filters from user preferences (stored from previous queries)
    existing_filters = user_preferences
    logger.debug(f" Existing filters from user preferences: {existing_filters}")
    # Build dynamic prompt based on actual schema - LLM handles everything
    prompt = _build_dynamic_extraction_prompt(conversation_context, user_preferences, settings, query, existing_filters)
    logger.debug(f" Prompt length: {len(prompt)} characters")
    logger.debug(f" Prompt preview: {prompt[:300]}...")
    raw = llm_client.generate(prompt).strip()
    logger.debug(f" LLM raw response: '{raw}'")
    llm_data = _json_from_text(raw)
    logger.debug(f" Parsed JSON: {llm_data}")
    # CRITICAL FIX: Merge with existing filters - new filters override existing ones
    # But preserve existing filters that are not overridden
    merged_filters = {**existing_filters, **llm_data}
    logger.debug(f" Merged filters: {merged_filters}")
    # DEBUG: Show what's being merged
    logger.debug(f" DEBUG MERGE:")
    logger.debug(f" Existing: {existing_filters}")
    logger.debug(f" New: {llm_data}")
    logger.debug(f" Final: {merged_filters}")
    # Process merged results with type coercion
    result = {}
    for key, value in merged_filters.items():
        if key not in allowed_fields:
            logger.debug(f" Skipping invalid field: {key}")
            continue
        ftype = field_types.get(key, "")
        normalized_value = _normalize_string_value(value, ftype, key)
        coerced = _coerce_value_by_type(normalized_value, ftype)
        if coerced is not None:
            result[key] = coerced
            logger.debug(f" Added filter: {key} = {coerced}")
    # Filter to only allowed fields
    result = {k: v for k, v in result.items() if k in allowed_fields}
    
    # Store the result in conversation for next query
    if hasattr(llm_client, 'conversation') and result:
        try:
            llm_client.conversation.update_preferences(result)
            logger.debug(f" Stored filters in conversation for next query")
        except Exception as e:
            logger.error(f" Error storing filters: {e}")
    logger.debug(f" Final filters: {result}")
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
    logger.debug(f" Conversation context extracted: {len(context)} characters")
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
            logger.debug(f" Context section added: {len(context_section)} characters")
            return final_prompt
        else:
            logger.warning(f" Warning: Could not find 'User query:' in prompt")
    logger.debug(f" No context section added")
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

    logger.debug(" FILTER EXTRACTION DEBUG:")
    logger.debug(f" Query: '{query}'")
    logger.debug(f" Extracted filters: {filters}")
    return query, filters
