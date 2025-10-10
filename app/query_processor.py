"""
Minimal query processor - uses LLM to extract filters.

Philosophy: LLM handles all extraction logic. This file just provides the prompt template.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
import json
import re
from .config import get_settings
from .utils import get_logger
from .instructions import LLM_FILTER_EXTRACTION_INSTRUCTIONS

logger = get_logger(__name__)


def extract_filters_from_query(query: str, llm_client) -> Dict[str, Any]:
    """
    Use LLM to extract filters from query.
    
    Simple approach:
    1. Build prompt with schema info
    2. Let LLM extract filters
    3. Return parsed JSON
    
    LLM handles ALL logic (no hardcoded patterns!)
    """
    logger.debug(f"🔍 QUERY PROCESSOR: Starting filter extraction for query: '{query}'")
    
    settings = get_settings()
    
    # Build schema description for LLM
    field_descriptions = _build_field_descriptions(settings)
    amenity_rules = _build_amenity_rules(settings)
    
    logger.debug(f"🔍 QUERY PROCESSOR: Built field descriptions ({len(field_descriptions)} chars)")
    logger.debug(f"🔍 QUERY PROCESSOR: Field descriptions:\n{field_descriptions}")
    
    # Get conversation context if available
    context = _get_context(llm_client)
    logger.debug(f"🔍 QUERY PROCESSOR: Conversation context length: {len(context)} chars")
    logger.debug(f"🔍 QUERY PROCESSOR: Conversation context: {context}")
    
    # Build prompt
    prompt = LLM_FILTER_EXTRACTION_INSTRUCTIONS.format(
        field_descriptions=field_descriptions,
        amenity_rules=amenity_rules,
        conversation_context=context,
        query=query
    )
    
    logger.debug(f"🔍 QUERY PROCESSOR: Filter extraction prompt length: {len(prompt)} chars")
    logger.debug(f"🔍 QUERY PROCESSOR: Full extraction prompt:\n{prompt}")
    
    # Let LLM extract
    try:
        logger.debug(f"🔍 QUERY PROCESSOR: Calling LLM for filter extraction...")
        response = llm_client.generate(prompt)
        logger.debug(f"🔍 QUERY PROCESSOR: LLM response length: {len(response)} chars")
        logger.debug(f"🔍 QUERY PROCESSOR: LLM raw response: {response}")
        
        filters = _parse_json(response)
        logger.debug(f"🔍 QUERY PROCESSOR: Successfully parsed filters: {filters}")
        
        # Store in conversation
        if hasattr(llm_client, 'conversation'):
            llm_client.conversation.update_preferences(filters)
            logger.debug(f"🔍 QUERY PROCESSOR: Updated conversation preferences")
        
        return filters
        
    except Exception as exc:
        logger.error(f"🔍 QUERY PROCESSOR: Filter extraction failed: {exc}")
        logger.error(f"🔍 QUERY PROCESSOR: Raw LLM response was: {response}")
        return {}


def preprocess_query(query: str, llm_client) -> Tuple[str, Dict[str, Any]]:
    """
    Preprocess query - extract filters using LLM.
    
    Returns:
        Tuple of (original_query, extracted_filters)
    """
    filters = extract_filters_from_query(query, llm_client)
    return query, filters


# ==================================================================================
# Helper Functions (Minimal)
# ==================================================================================

def _build_field_descriptions(settings) -> str:
    """Build field schema description for LLM"""
    allowed_fields = settings.retrieval.filter_fields or []
    field_types = settings.database.field_types or {}
    
    lines = []
    for field in allowed_fields:
        ftype = field_types.get(field, "keyword")
        
        if ftype in ("integer", "float"):
            lines.append(f"- {field}: number or range {{'gte': X, 'lte': Y}}")
        elif ftype == "boolean":
            lines.append(f"- {field}: true/false")
        else:
            lines.append(f"- {field}: string")
    
    return "\n".join(lines)


def _build_amenity_rules(settings) -> str:
    """Build amenity extraction rules from config"""
    boolean_fields = settings.database.boolean_fields or {}
    
    if not boolean_fields:
        return ""
    
    rules = ["Amenities (set to true if mentioned):"]
    for field_name, display_label in boolean_fields.items():
        rules.append(f"- {display_label} → {field_name}")
    
    return "\n".join(rules)


def _get_context(llm_client) -> str:
    """Get conversation context if available"""
    if not hasattr(llm_client, 'conversation'):
        return ""
    
    messages = llm_client.conversation.get_messages()[-6:]  # Last 3 exchanges
    
    context_parts = []
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            content = msg["content"][:100]
            if len(msg["content"]) > 100:
                content += "..."
            context_parts.append(f"{msg['role']}: {content}")
    
    return " | ".join(context_parts)


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response"""
    try:
        # Try direct parse
        return json.loads(text)
    except:
        pass
    
    # Try to extract JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    # Find nested JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    return {}
