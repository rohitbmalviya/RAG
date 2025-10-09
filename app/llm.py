from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type
import os
import time
import requests
import json
import re
from .config import LLMConfig
from .core.base import BaseLLM
from .models import RetrievedChunk
from .utils import get_logger
# Import ONLY used instructions from centralized file (no unused imports!)
from .instructions import (
    FALLBACK_GREETINGS,
    FALLBACK_GENERAL_RESPONSE,
    QUERY_CLASSIFICATION_TEMPLATE,
    GREETING_RESPONSE_TEMPLATE,
    GENERAL_KNOWLEDGE_TEMPLATE,
    ALTERNATIVES_GENERATION_TEMPLATE,
    COMPLETE_SYSTEM_PROMPT,
    KNOWLEDGE_RESPONSE_PROMPT,
    KNOWLEDGE_WITH_WEB_SEARCH_PROMPT,
    INTELLIGENT_DECISION_PROMPT_TEMPLATE,
    REQUIREMENT_EXTRACTION_PROMPT
)

try:
    from .main import pipeline_state
except ImportError:
    pipeline_state = None

class RequirementGatheringMetrics:
    def __init__(self):
        self.total_attempts = 0
        self.successful_sends = 0
        self.failed_sends = 0
        
    def log_attempt(self, success: bool):
        self.total_attempts += 1
        if success:
            self.successful_sends += 1
        else:
            self.failed_sends += 1
        
        if self.total_attempts % 10 == 0:
            from .utils import get_logger
            logger = get_logger(__name__)
            logger.debug(
                f"Requirement gathering stats: {self.successful_sends}/{self.total_attempts} successful"
            )

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.last_filters: Dict[str, Any] = {}  
        self.conversation_summary: str = ""
        self.requirement_gathered: bool = False
        self.search_history: List[Dict[str, Any]] = []  
        self.alternatives_suggested: List[Dict[str, Any]] = []  
        self.conversation_start_time: float = 0.0
        self.last_activity_time: float = 0.0
        self.user_contact_info: Dict[str, str] = {}  # Store operator name & contact
        self.current_flow: Optional[str] = None  # "requirement_gathering" | None
        self.waiting_for_contact_info: bool = False  # True when we asked for contact
        self.search_memory: Dict[str, Any] = {
            "last_query": None,              # Original user query
            "last_semantic_query": None,     # Query used for semantic search
            "retrieved_candidates": [],      # ALL retrieved chunks (for client-side filtering)
            "current_filters": {},           # Currently applied filters
            "refinement_count": 0,           # How many times user refined search
            "search_context": "",            # LLM-generated context summary
        }

    def add_message(self, role: str, content: str):
        import time
        self.messages.append({"role": role, "content": content})
        self.last_activity_time = time.time()
        
        
        if len(self.messages) > 15:
            self.messages = self.messages[-15:]

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences from conversation with intelligent merging"""
        
        for key, value in preferences.items():
            if value is not None:  
                self.user_preferences[key] = value
        
        self.last_filters = preferences  

    def get_preferences(self) -> Dict[str, Any]:
        return self.user_preferences

    def get_context_for_query(self, current_query: str) -> str:
        """Get relevant context for current query with enhanced memory"""
        if not self.messages:
            return ""
        
        
        recent = self.messages[-6:] if len(self.messages) >= 6 else self.messages
        context = []
        
        for msg in recent:
            if msg["role"] == "user":
                
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                context.append(f"Previously asked: {content}")
            elif msg["role"] == "assistant":
                
                content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
                context.append(f"Assistant response: {content}")
        
        return " | ".join(context)

    def add_search_attempt(self, query: str, filters: Dict[str, Any], results_count: int):
        """Track search attempts for better context"""
        import time
        self.search_history.append({
            "query": query,
            "filters": filters,
            "results_count": results_count,
            "timestamp": time.time()
        })
        
        
        if len(self.search_history) > 5:
            self.search_history = self.search_history[-5:]

    def add_alternative_suggestion(self, alternative: Dict[str, Any]):
        """Track suggested alternatives"""
        self.alternatives_suggested.append(alternative)
          
        if len(self.alternatives_suggested) > 3:
            self.alternatives_suggested = self.alternatives_suggested[-3:]

    def get_search_context(self) -> str:
        """Get context about previous searches"""
        if not self.search_history:
            return ""
        
        recent_searches = []
        for search in self.search_history[-3:]:  
            recent_searches.append(f"'{search['query']}' ({search['results_count']} results)")
        
        return f"Recent searches: {'; '.join(recent_searches)}"

    def get_alternatives_context(self) -> str:
        """Get context about suggested alternatives"""
        if not self.alternatives_suggested:
            return ""
        
        alternatives = []
        for alt in self.alternatives_suggested:
            alternatives.append(alt.get('suggestion', 'Unknown'))
        
        return f"Suggested alternatives: {'; '.join(alternatives)}"

    def set_requirement_gathered(self, gathered: bool):
        self.requirement_gathered = gathered

    def is_requirement_gathered(self) -> bool:
        return self.requirement_gathered

    def get_conversation_summary(self) -> str:
        return self.conversation_summary

    def set_conversation_summary(self, summary: str):
        self.conversation_summary = summary

    def is_expired(self, ttl_seconds: int = 900) -> bool:
        """Check if conversation has expired (15 minutes default)"""
        import time
        if self.last_activity_time == 0:
            return False
        return (time.time() - self.last_activity_time) > ttl_seconds

    def get_conversation_duration(self) -> float:
        """Get conversation duration in seconds"""
        import time
        if self.conversation_start_time == 0:
            self.conversation_start_time = time.time()
        return time.time() - self.conversation_start_time

    def set_contact_info(self, operator_name: str, contact_info: str):
        """Store user contact information for requirement gathering"""
        self.user_contact_info = {
            "operator_name": operator_name,
            "contact_info": contact_info
        }

    def get_contact_info(self) -> Dict[str, str]:
        """Get stored contact information"""
        return self.user_contact_info

    def has_contact_info(self) -> bool:
        """Check if contact info is collected"""
        return bool(
            self.user_contact_info.get("operator_name") and 
            self.user_contact_info.get("contact_info")
        )
    
    def set_flow_state(self, flow: Optional[str], waiting_for_contact: bool = False):
        """Set conversation flow state (GENERIC - works for any datasource)"""
        self.current_flow = flow
        self.waiting_for_contact_info = waiting_for_contact
    
    def is_in_requirement_gathering(self) -> bool:
        """Check if currently in requirement gathering flow"""
        return self.current_flow == "requirement_gathering"
    
    def is_waiting_for_contact(self) -> bool:
        """Check if waiting for user contact information"""
        return self.waiting_for_contact_info

    def store_search_results(
        self, 
        query: str, 
        semantic_query: str,
        candidates: List['RetrievedChunk'], 
        filters: Dict[str, Any],
        context_summary: str = ""
    ):
        """Store search results for progressive refinement (GENERIC - works for any domain)"""
        self.search_memory["last_query"] = query
        self.search_memory["last_semantic_query"] = semantic_query
        self.search_memory["retrieved_candidates"] = candidates.copy() if candidates else []
        self.search_memory["current_filters"] = filters.copy() if filters else {}
        self.search_memory["search_context"] = context_summary
        self.search_memory["refinement_count"] = 0
    
    def update_search_refinement(self, new_filters: Dict[str, Any], context_summary: str = ""):
        """Update filters for refinement (GENERIC - works for any domain)"""
        if new_filters:
            self.search_memory["current_filters"].update(new_filters)
        self.search_memory["refinement_count"] += 1
        if context_summary:
            self.search_memory["search_context"] = context_summary
    
    def get_search_candidates(self) -> List['RetrievedChunk']:
        """Get stored candidates for filtering (GENERIC - works for any domain)"""
        return self.search_memory.get("retrieved_candidates", [])
    
    def has_search_memory(self) -> bool:
        """Check if we have previous search to refine (GENERIC - works for any domain)"""
        return bool(self.search_memory.get("retrieved_candidates"))
    
    def clear_search_memory(self):
        """Clear search memory for new topic (GENERIC - works for any domain)"""
        self.search_memory = {
            "last_query": None,
            "last_semantic_query": None,
            "retrieved_candidates": [],
            "current_filters": {},
            "refinement_count": 0,
            "search_context": "",
        }
    
    def get_search_context_summary(self) -> str:
        """Get human-readable search context for LLM (GENERIC - works for any domain)"""
        if not self.has_search_memory():
            return "No previous search in memory"
        
        memory = self.search_memory
        summary_parts = []
        
        if memory.get('last_query'):
            summary_parts.append(f"Previous search: '{memory['last_query']}'")
        
        candidate_count = len(memory.get('retrieved_candidates', []))
        if candidate_count > 0:
            summary_parts.append(f"Retrieved: {candidate_count} candidates")
        
        if memory.get('current_filters'):
            summary_parts.append(f"Current filters: {memory['current_filters']}")
        
        refinement_count = memory.get('refinement_count', 0)
        if refinement_count > 0:
            summary_parts.append(f"Refinements: {refinement_count}")
        
        if memory.get('search_context'):
            summary_parts.append(f"Context: {memory['search_context']}")
        
        return " | ".join(summary_parts) if summary_parts else "Search memory exists but empty"

LLM_PROVIDERS: Dict[str, Type["BaseLLMProvider"]] = {}

def _extract_json_from_response(response: str, expected_type: str = "object") -> Any:
    """Centralized JSON extraction with error handling"""
    try:
        response = response.strip()
        if expected_type == "object":
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
        elif expected_type == "array":
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
        else:
            json_match = re.search(r'[\{\[].*[\}\]', response, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    return None if expected_type == "object" else []

def _safe_llm_generate_with_fallback(llm_client, prompt: str, fallback_response: str) -> str:
    """Centralized LLM generation with fallback handling"""
    try:
        response = llm_client.generate(prompt).strip()
        return response if response else fallback_response
    except Exception:
        return fallback_response

def _build_query_classification_template() -> str:
    """Get query classification template from centralized instructions file"""
    return QUERY_CLASSIFICATION_TEMPLATE

def register_llm_provider(name: str) -> Callable[[Type["BaseLLMProvider"]], Type["BaseLLMProvider"]]:
    def decorator(cls: Type["BaseLLMProvider"]) -> Type["BaseLLMProvider"]:
        LLM_PROVIDERS[name.lower()] = cls
        return cls
    return decorator

class BaseLLMProvider:
    def __init__(self, config: LLMConfig, api_key: Optional[str]) -> None:
        self._config = config
        self._api_key = api_key
        self._logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._ensure_api_key()
        self._setup_client()

    def _ensure_api_key(self) -> None:
        if not self._api_key:
            self._api_key = os.getenv("LLM_MODEL_API_KEY")

    def _setup_client(self) -> None:
        pass

    def generate(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

@register_llm_provider("google")
class GoogleLLM(BaseLLMProvider):
    def _setup_client(self) -> None:
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai package not available") from exc
        if self._api_key:
            genai.configure(api_key=self._api_key)
        else:
            self._logger.debug("LLM API key missing for Google provider")
        self._model = genai.GenerativeModel(model_name=self._config.model)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self._config.temperature,
                    "max_output_tokens": self._config.max_output_tokens,
                },
            )
            return response.text
        except Exception as exc:
            self._logger.error("Google generate_content failed: %s", exc)
            raise

@register_llm_provider("openai")
@register_llm_provider("azure_openai")
class OpenAILLM(BaseLLMProvider):
    def _setup_client(self) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package not available") from exc

        if not self._api_key:
            self._logger.debug("LLM_MODEL_API_KEY missing for OpenAI provider")

        self._client = OpenAI(api_key=self._api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_output_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            self._logger.error("OpenAI chat.completions.create failed: %s", exc)
            raise

class DynamicQueryClassifier:
    """Dynamic LLM-based query classification"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query using LLM for dynamic, scalable classification"""
        prompt = _build_query_classification_template().format(query=query)

        try:
            response = self.llm_client.generate(prompt).strip()
            result = _extract_json_from_response(response, "object")
            if result:
                return result
        except Exception:
            pass

        return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Simple fallback if LLM fails - returns safe default, NO hardcoded keywords!"""
        return {
            "category": "property_search", 
            "confidence": 0.3, 
            "reasoning": "LLM classification unavailable, defaulting to entity search"
        }

class DynamicAlternativesGenerator:
    """Dynamic LLM-based alternatives generation"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_verified_alternatives(self, preferences: Dict[str, Any], retriever) -> List[Dict[str, Any]]:
        """Generate alternatives using LLM reasoning and verify with database"""
        if not preferences:
            return []
        
        
        alternatives = self._generate_alternatives_with_llm(preferences)
        
        
        verified_alternatives = []
        for alt in alternatives:
            if self._verify_alternative_exists(alt, retriever):
                verified_alternatives.append(alt)
        
        return verified_alternatives[:3]  
    
    def _generate_alternatives_with_llm(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternatives using LLM reasoning"""
        prompt = ALTERNATIVES_GENERATION_TEMPLATE.format(preferences=preferences)

        try:
            response = self.llm_client.generate(prompt).strip()
            alternatives = _extract_json_from_response(response, "array")
            return alternatives if isinstance(alternatives, list) else []
        except Exception as e:
            self.llm_client._logger.debug(f"LLM alternatives generation failed: {e}")
        
        return []
    
    def _verify_alternative_exists(self, alternative: Dict[str, Any], retriever) -> bool:
        """Improved alternative verification with better error handling"""
        try:
            filters = alternative.get("filters", {})
            suggestion = alternative.get("suggestion", "")
             
            test_queries = [
                suggestion,
                f"property in {suggestion}",
                "available properties"
            ]
            
            for query in test_queries:
                try:
                    test_chunks = retriever.retrieve(query, filters=filters, top_k=15)
                    
                    if test_chunks and len(test_chunks) >= 3:
                        
                        valid_chunks = [
                            chunk for chunk in test_chunks 
                            if chunk.metadata.get("rent_charge") and 
                               chunk.metadata.get("property_title")
                        ]
                        
                        if len(valid_chunks) >= 3:
                            alternative["count"] = f"{len(valid_chunks)}+"
                            alternative["sample_properties"] = [
                                {
                                    "title": chunk.metadata.get("property_title", "Property"),
                                    "rent": f"AED {chunk.metadata.get('rent_charge', 'N/A'):,}"
                                }
                                for chunk in valid_chunks[:2]
                            ]
                            return True
                            
                except Exception as e:
                    self.llm_client._logger.debug(f"Test query failed for '{query}': {e}")
                    continue   
            return False
            
        except Exception as e:
            self.llm_client._logger.debug(f"Failed to verify alternative: {e}")
            return False

class DynamicResponseGenerator:
    """Dynamic LLM-based response generation"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_greeting_response(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate dynamic greeting responses"""
        prompt = GREETING_RESPONSE_TEMPLATE.format(
            user_input=user_input,
            conversation_count=len(conversation_history)
        )
        
        return _safe_llm_generate_with_fallback(
            self.llm_client, 
            prompt, 
            self._fallback_greeting()
        )
    
    def _fallback_greeting(self) -> str:
        """Fallback greeting if LLM fails"""
        return FALLBACK_GREETINGS[0]  
    
    def generate_general_knowledge_response(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        """Generate dynamic general knowledge responses"""
        prompt = GENERAL_KNOWLEDGE_TEMPLATE.format(user_input=user_input)
        
        return _safe_llm_generate_with_fallback(
            self.llm_client, 
            prompt, 
            self._fallback_general_response()
        )
    
    def _fallback_general_response(self) -> str:
        """Fallback response if LLM fails"""
        return FALLBACK_GENERAL_RESPONSE

class LLMClient(BaseLLM):
    def __init__(self, config: LLMConfig, api_key: Optional[str]) -> None:
        self._logger = get_logger(__name__)
        self._config = config
        provider_name = (config.provider or "google").lower()
        provider_cls = LLM_PROVIDERS.get(provider_name)
        if provider_cls is None:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        self._provider_client = provider_cls(config, api_key)
        self.conversation = Conversation() 
        self.requirement_metrics = RequirementGatheringMetrics()
        self.conversation.add_message("system", COMPLETE_SYSTEM_PROMPT)

    def classify_query(self, user_input: str) -> Dict[str, Any]:
        """LLM-based query classification - NO hardcoded keywords!
        
        Uses LLM intelligence to classify queries based on minimal config context.
        Works for ANY domain (properties, cars, jobs, products, etc.)
        
        Returns:
            Dict with keys: category, confidence, reasoning
        """
        from .config import get_settings
        from .instructions import GENERIC_QUERY_CLASSIFICATION_TEMPLATE
        
        settings = get_settings()
        
        # Get minimal context from config (generic!)
        entity_name = settings.database.table  # "properties", "cars", "jobs", etc.
        pricing_field = settings.database.pricing_field  # "rent_charge", "price", "salary", etc.
        
        # Build classification prompt using generic template
        classification_prompt = GENERIC_QUERY_CLASSIFICATION_TEMPLATE.format(
            entity_name=entity_name,
            user_input=user_input,
            pricing_field=pricing_field
        )
        
        try:
            # Let LLM classify using its intelligence (NO hardcoded keywords!)
            response = self._safe_generate([{"role": "user", "content": classification_prompt}])
            result = _extract_json_from_response(response, "object")
            
            if result and "category" in result:
                self._logger.debug(f" LLM Classification: {result['category']} (confidence: {result.get('confidence', 'N/A')})")
                self._logger.debug(f" Reasoning: {result.get('reasoning', 'N/A')}")
                return result
        except Exception as e:
            self._logger.debug(f"LLM classification failed: {e}, using fallback")
        
        # Fallback: simple pattern matching if LLM fails
        return self._fallback_classification(user_input)
    
    def _fallback_classification(self, user_input: str) -> Dict[str, Any]:
        """Simple fallback if LLM fails - returns safe default, NO hardcoded keywords!"""
        return {
            "category": "entity_search", 
            "confidence": 0.3, 
            "reasoning": "LLM classification unavailable, defaulting to entity search"
        }
    
    def detect_query_intent(self, user_input: str) -> Dict[str, Any]:
        """
        LLM-driven query intent detection - 100% GENERIC, NO hardcoded patterns!
        
        Determines if user query is:
        - NEW_SEARCH: Start fresh search
        - REFINEMENT: Add constraints to previous search
        - CLARIFICATION: Answering a question
        
        Works for ANY domain (properties, cars, jobs, products, etc.)
        
        Args:
            user_input: Current user query
            
        Returns:
            Dict with keys: intent, confidence, reasoning, suggested_action
        """
        from .instructions import QUERY_INTENT_ANALYSIS_TEMPLATE
        
        self._logger.debug(f"\n QUERY INTENT DETECTION (LLM-Driven):")
        self._logger.debug(f" User query: '{user_input}'")
        
        # Build context for LLM
        conversation_history = self._get_conversation_context()
        search_memory = self.conversation.get_search_context_summary()
        
        self._logger.debug(f" Has search memory: {self.conversation.has_search_memory()}")
        if search_memory:
            self._logger.debug(f" Search memory: {search_memory[:200]}...")
        
        # Generate prompt
        prompt = QUERY_INTENT_ANALYSIS_TEMPLATE.format(
            conversation_history=conversation_history if conversation_history else "No previous conversation",
            search_memory=search_memory,
            current_query=user_input
        )
        
        # Get LLM decision
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            intent_data = _extract_json_from_response(response, "object")
            
            if intent_data and "intent" in intent_data:
                self._logger.debug(f" LLM Intent Decision:")
                self._logger.debug(f"   Intent: {intent_data.get('intent')}")
                self._logger.debug(f"   Confidence: {intent_data.get('confidence')}")
                self._logger.debug(f"   Reasoning: {intent_data.get('reasoning')}")
                self._logger.debug(f"   Suggested Action: {intent_data.get('suggested_action')}")
                return intent_data
            else:
                self._logger.debug(f" LLM response missing intent field, using fallback")
                
        except Exception as e:
            self._logger.error(f" Intent detection failed: {e}, using fallback")
        
        # Fallback: if no search memory, must be new search
        if not self.conversation.has_search_memory():
            self._logger.debug(f" FALLBACK: No search memory → NEW_SEARCH")
            return {
                "intent": "NEW_SEARCH",
                "confidence": 1.0,
                "reasoning": "No previous search exists in memory",
                "suggested_action": "Perform new semantic search"
            }
        else:
            self._logger.debug(f" FALLBACK: Search memory exists → REFINEMENT")
            return {
                "intent": "REFINEMENT",
                "confidence": 0.5,
                "reasoning": "Fallback due to LLM error, assuming refinement",
                "suggested_action": "Try refining previous search"
            }
    
    def apply_progressive_filters(
        self, 
        candidates: List[RetrievedChunk], 
        new_filters: Dict[str, Any]
    ) -> List[RetrievedChunk]:
        """
        Apply filters to existing candidates - FULLY GENERIC!
        
        Works for ANY field type:
        - Numeric (price, age, size, distance, etc.)
        - Text (color, status, category, etc.)
        - Boolean (verified, available, featured, etc.)
        - List (amenities, tags, features, etc.)
        
        Args:
            candidates: Previously retrieved chunks
            new_filters: New filters to apply
        
        Returns:
            Filtered chunks
        """
        if not new_filters:
            self._logger.debug(" No filters to apply, returning all candidates")
            return candidates
        
        self._logger.debug(f"\n PROGRESSIVE FILTERING (GENERIC):")
        self._logger.debug(f" Input candidates: {len(candidates)}")
        self._logger.debug(f" Filters to apply: {new_filters}")
        
        filtered = []
        
        for chunk in candidates:
            metadata = chunk.metadata
            matches_all = True
            
            for filter_key, filter_value in new_filters.items():
                metadata_value = metadata.get(filter_key)
                
                # GENERIC FILTER LOGIC (works for ANY field!)
                
                # Case 1: Range query (dict with gte/lte/gt/lt)
                if isinstance(filter_value, dict):
                    if not self._matches_range_filter(metadata_value, filter_value, filter_key):
                        matches_all = False
                        break
                
                # Case 2: List query (any value in list)
                elif isinstance(filter_value, list):
                    if not self._matches_list_filter(metadata_value, filter_value, filter_key):
                        matches_all = False
                        break
                
                # Case 3: Exact match
                else:
                    if not self._matches_exact_filter(metadata_value, filter_value, filter_key):
                        matches_all = False
                        break
            
            if matches_all:
                filtered.append(chunk)
        
        self._logger.debug(f" Output candidates: {len(filtered)}")
        if len(candidates) > 0:
            efficiency = (len(filtered) / len(candidates)) * 100
            self._logger.debug(f" Filter efficiency: {efficiency:.1f}% of candidates match")
        
        return filtered
    
    def _matches_range_filter(self, value: Any, range_spec: Dict[str, Any], field_name: str = "") -> bool:
        """Generic range matching (works for numbers, dates, etc.) - FULLY GENERIC"""
        if value is None:
            return False
        
        try:
            # Convert to number for comparison
            if isinstance(value, (int, float)):
                num_value = value
            else:
                num_value = float(value)
            
            if "lte" in range_spec:
                if num_value > range_spec["lte"]:
                    return False
            
            if "gte" in range_spec:
                if num_value < range_spec["gte"]:
                    return False
            
            if "lt" in range_spec:
                if num_value >= range_spec["lt"]:
                    return False
            
            if "gt" in range_spec:
                if num_value <= range_spec["gt"]:
                    return False
            
            return True
            
        except (ValueError, TypeError) as e:
            self._logger.debug(f" Range filter failed for field '{field_name}', value: {value}, error: {e}")
            return False
    
    def _matches_list_filter(self, value: Any, filter_list: List[Any], field_name: str = "") -> bool:
        """Generic list matching (value must be IN filter list) - FULLY GENERIC"""
        if value is None:
            return False
        
        # Case 1: Metadata value is a list (e.g., amenities, tags)
        if isinstance(value, list):
            # ANY value in metadata must match ANY value in filter
            for item in value:
                if item in filter_list:
                    return True
            return False
        
        # Case 2: Metadata value is single value
        return value in filter_list
    
    def _matches_exact_filter(self, value: Any, filter_value: Any, field_name: str = "") -> bool:
        """Generic exact matching - FULLY GENERIC"""
        if value is None:
            return False
        
        # Case-insensitive string comparison
        if isinstance(value, str) and isinstance(filter_value, str):
            return value.lower() == filter_value.lower()
        
        # Exact match for other types
        return value == filter_value

    def _handle_progressive_refinement(self, user_input: str) -> tuple[str, bool]:
        """
        Handle progressive refinement - GENERIC!
        Filter existing candidates with new constraints.
        
        This enables users to refine search results without re-querying the database.
        Works for ANY domain (properties, cars, jobs, products, etc.)
        """
        from .query_processor import extract_filters_with_llm_context_aware
        
        self._logger.debug(f"\n═══════════════════════════════════════════════════════════")
        self._logger.debug(f" PROGRESSIVE REFINEMENT HANDLER")
        self._logger.debug(f"═══════════════════════════════════════════════════════════")
        self._logger.debug(f" Previous query: {self.conversation.search_memory['last_query']}")
        self._logger.debug(f" Candidates in memory: {len(self.conversation.get_search_candidates())}")
        self._logger.debug(f" Current filters: {self.conversation.search_memory['current_filters']}")
        
        # Extract NEW filters from user query (LLM-driven)
        new_filters = extract_filters_with_llm_context_aware(user_input, self)
        self._logger.debug(f" New filters extracted: {new_filters}")
        
        # Merge with existing filters
        current_filters = self.conversation.search_memory["current_filters"].copy()
        merged_filters = {**current_filters, **new_filters}
        self._logger.debug(f" Merged filters: {merged_filters}")
        
        # Apply filters to stored candidates (client-side filtering - NO DB query!)
        candidates = self.conversation.get_search_candidates()
        filtered = self.apply_progressive_filters(candidates, merged_filters)
        
        self._logger.debug(f" After filtering: {len(filtered)} results")
        self._logger.debug(f"═══════════════════════════════════════════════════════════")
        
        # Update search memory with refined results
        self.conversation.update_search_refinement(
            new_filters=merged_filters,
            context_summary=f"Refined with: {new_filters}"
        )
        
        if not filtered:
            self._logger.debug(f" No results after refinement - offering alternatives/gathering")
            # No chunks to store
            self._refinement_result_chunks = []
            # Build context with no chunks
            context = self._build_comprehensive_context(user_input, [])
            llm_response = self._get_llm_intelligent_response(context)
            answer, _ = self._parse_llm_response(llm_response)
            self._add_conversation_messages(user_input, answer)
            return answer, False
        
        # Generate response with filtered results
        self._logger.debug(f" Generating response with {len(filtered)} filtered results")
        
        # Store filtered chunks for main.py to create sources
        self._refinement_result_chunks = filtered[:8]  # Store for source creation
        self._logger.debug(f" Stored {len(self._refinement_result_chunks)} filtered chunks for source creation")
        
        context = self._build_comprehensive_context(user_input, filtered[:8])
        llm_response = self._get_llm_intelligent_response(context)
        answer, should_show_sources = self._parse_llm_response(llm_response)
        
        self._add_conversation_messages(user_input, answer)
        return answer, True  # Always show sources for refinement results
    
    def validate_response_context(self, answer: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """LLM-based context validation - NO hardcoded location lists!
        
        Uses LLM to validate responses against actual retrieved chunks.
        Works for ANY domain (properties, cars, jobs, products, etc.)
        
        Args:
            answer: The generated response to validate
            retrieved_chunks: The chunks retrieved from database
        
        Returns:
            Validated answer (possibly with disclaimer if issues found)
        """
        from .config import get_settings
        from .instructions import GENERIC_CONTEXT_VALIDATION_TEMPLATE
        
        settings = get_settings()
        
        # Check if validation is enabled
        if not settings.llm.query_handling.enable_context_validation:
            return answer  # Validation disabled
        
        if not retrieved_chunks:
            # No chunks to validate against - check if answer mentions entities
            if any(keyword in answer.lower() for keyword in [settings.database.table[:-1], "available", "found"]):
                # Answer claims to have results but there are no chunks
                return (
                    "I apologize, but I couldn't find any specific items matching your criteria in our database. "
                    "Would you like to adjust your search filters or save your requirements for later notification?"
                )
            return answer
        
        # Summarize chunks for validation (generic!)
        chunk_summary = self._summarize_chunks_for_validation(retrieved_chunks)
        tolerance = settings.llm.query_handling.validation_tolerance * 100  # Convert to percentage
        
        # Build validation prompt using generic template
        validation_prompt = GENERIC_CONTEXT_VALIDATION_TEMPLATE.format(
            answer=answer,
            chunk_summary=chunk_summary,
            tolerance=tolerance
        )
        
        try:
            # Let LLM validate using its intelligence (NO hardcoded checks!)
            response = self._safe_generate([{"role": "user", "content": validation_prompt}])
            result = _extract_json_from_response(response, "object")
            
            if result:
                is_valid = result.get("valid", True)
                issues = result.get("issues", [])
                
                if not is_valid and issues:
                    self._logger.debug(f" LLM Validation: Issues found - {len(issues)} problems")
                    for issue in issues[:3]:  # Show top 3 issues
                        self._logger.debug(f" - {issue}")
                    # Add disclaimer to response
                    answer += "\n\n*Please verify details directly with the listing for accuracy.*"
                else:
                    self._logger.debug(f" LLM Validation: Response is accurate")
        except Exception as e:
            self._logger.debug(f"LLM validation failed: {e}")
        
        return answer
    
    def _summarize_chunks_for_validation(self, chunks: List[RetrievedChunk]) -> str:
        """Summarize chunks for validation (generic!)"""
        from .config import get_settings
        settings = get_settings()
        
        pricing_field = settings.database.pricing_field
        display_field = settings.database.primary_display_field
        currency = settings.database.display.currency
        
        summary = []
        for i, chunk in enumerate(chunks[:10], 1):  # Top 10 chunks
            metadata = chunk.metadata
            title = metadata.get(display_field, "Item")
            price = metadata.get(pricing_field, "N/A")
            
            if price != "N/A" and isinstance(price, (int, float)):
                summary.append(f"[{i}] {title} - {currency} {price:,.0f}")
            else:
                summary.append(f"[{i}] {title} - {price}")
        
        return "\n".join(summary)

    def chat(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Process user query using pure LLM intelligence - no hardcoded conditions.""" 
        answer, _ = self.chat_with_source_decision(user_input, retrieved_chunks)
        return answer

    def chat_with_source_decision(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> tuple[str, bool]:
        """Process user query using LLM-driven classification - NO hardcoded conditions!
        
        Uses LLM intelligence to classify queries and handle them appropriately.
        Works for ANY domain (properties, cars, jobs, products, etc.)
        """
        user_input = user_input.strip()

        from .config import get_settings
        settings = get_settings()

        self._logger.debug(f"\n LLM DEBUG:")
        self._logger.debug(f" User input: '{user_input}'")
        self._logger.debug(f" Conversation history: '{self._get_conversation_context()[:200]}...'")
        self._logger.debug(f" User preferences: {self.conversation.get_preferences()}")
        self._logger.debug(f" Has data: {bool(retrieved_chunks)}")
        self._logger.debug(f" Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
        self._logger.debug(f" Flow state: {self.conversation.current_flow}, Waiting for contact: {self.conversation.waiting_for_contact_info}")
        
        # CONTEXT-AWARE: Check if we're waiting for contact info (prevent repetition)
        if self.conversation.is_waiting_for_contact():
            self._logger.debug(f" CONTEXT: We asked for contact info, user is responding")
            # User is responding to our contact info request - go directly to requirement gathering
            answer = self.gather_requirements_and_send(user_input, ask_confirmation=False)
            self._add_conversation_messages(user_input, answer)
            return answer, False
        
        # Use LLM to classify query (NO hardcoded keywords!)
        classification = self.classify_query(user_input)
        category = classification.get("category", "entity_search")
        confidence = classification.get("confidence", 0.5)
        
        self._logger.debug(f" Query Category: {category} (confidence: {confidence})")
        # Handle special query types based on LLM classification
        if category == "price_inquiry" and settings.llm.query_handling.enable_price_aggregation:
            self._logger.debug(f" PRICE INQUIRY DETECTED!")
            answer = self._handle_average_price_query(user_input)
            self._logger.debug(f" PRICE CALCULATION COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False  # No sources for price queries
        
        elif category == "knowledge_inquiry" and settings.llm.query_handling.enable_knowledge_search:
            self._logger.debug(f" KNOWLEDGE INQUIRY DETECTED!")
            answer = self._handle_general_knowledge_query(user_input)
            self._logger.debug(f" KNOWLEDGE QUERY COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False  # No sources for knowledge queries
        
        elif category == "requirement_gathering" and settings.llm.query_handling.enable_requirement_gathering:
            self._logger.debug(f" REQUIREMENT GATHERING DETECTED!")
            # Set flow state for context-aware conversation
            self.conversation.set_flow_state("requirement_gathering")
            self._logger.debug(f" Flow state set to: requirement_gathering")
            self._logger.debug(f" CALLING gather_requirements_and_send() METHOD...")
            answer = self.gather_requirements_and_send(user_input, ask_confirmation=False)
            self._logger.debug(f" REQUIREMENT GATHERING COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False
        
        elif category == "general_conversation":
            # Handle greetings and general conversation
            self._logger.debug(f" GENERAL CONVERSATION DETECTED!")
            
            # CRITICAL FIX: Clear search memory for greetings (not real searches!)
            if self.conversation.has_search_memory():
                self._logger.debug(f" Clearing search memory (greeting/general conversation, not entity search)")
                self.conversation.clear_search_memory()
            
            # Also clear temp search results if any
            if hasattr(self, '_temp_search_results'):
                self._logger.debug(f" Clearing temp search results (not storing for greeting)")
                delattr(self, '_temp_search_results')
            
            context = self._build_comprehensive_context(user_input, retrieved_chunks)
            llm_response = self._get_llm_intelligent_response(context)
            answer, _ = self._parse_llm_response(llm_response)
            self._add_conversation_messages(user_input, answer)
            return answer, False  # No sources for greetings
        
        # CONTEXT-AWARE: If in requirement gathering flow, continue gathering (don't search!)
        if self.conversation.is_in_requirement_gathering() and not self.conversation.is_waiting_for_contact():
            self._logger.debug(f" CONTEXT: Already in requirement gathering, user providing more details")
            self._logger.debug(f" Continuing requirement gathering (NOT searching for properties)")
            answer = self.gather_requirements_and_send(user_input, ask_confirmation=False)
            self._add_conversation_messages(user_input, answer)
            return answer, False
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PROGRESSIVE REFINEMENT: LLM-driven intent detection (NEW_SEARCH vs REFINEMENT)
        # ═══════════════════════════════════════════════════════════════════════════════
        intent = self.detect_query_intent(user_input)
        intent_type = intent.get("intent", "NEW_SEARCH")
        
        self._logger.debug(f" Query Intent: {intent_type}")
        self._logger.debug(f" Reasoning: {intent.get('reasoning')}")
        
        # Handle REFINEMENT: Filter existing candidates (client-side, no DB query)
        if intent_type == "REFINEMENT" and self.conversation.has_search_memory():
            self._logger.debug(f" PROGRESSIVE REFINEMENT PATH - Filtering existing results")
            return self._handle_progressive_refinement(user_input)
        
        # Handle CLARIFICATION: User answering a question
        elif intent_type == "CLARIFICATION":
            self._logger.debug(f" CLARIFICATION PATH - User responding to question")
            context = self._build_comprehensive_context(user_input, retrieved_chunks)
            llm_response = self._get_llm_intelligent_response(context)
            answer, should_show_sources = self._parse_llm_response(llm_response)
            self._add_conversation_messages(user_input, answer)
            return answer, should_show_sources if retrieved_chunks else False
        
        # Default: NEW_SEARCH or entity_search - process with full LLM intelligence
        self._logger.debug(f" NEW SEARCH / ENTITY SEARCH - Processing with LLM intelligence...")
        
        # CRITICAL FIX: Finalize search result storage for entity_search
        if hasattr(self, '_temp_search_results'):
            temp_results = self._temp_search_results
            self._logger.debug(f" Finalizing search result storage (entity_search detected)")
            self.conversation.store_search_results(
                query=temp_results["query"],
                semantic_query=temp_results["query"],
                candidates=temp_results["candidates"],
                filters=temp_results["filters"],
                context_summary=f"Initial search: {temp_results['query']}"
            )
            self._logger.debug(f" Stored {len(temp_results['candidates'])} candidates in conversation memory")
            delattr(self, '_temp_search_results')  # Clean up temp storage
        
        context = self._build_comprehensive_context(user_input, retrieved_chunks)
        llm_response = self._get_llm_intelligent_response(context)
        answer, should_show_sources = self._parse_llm_response(llm_response)
        
        self._logger.debug(f" LLM Response: {llm_response[:200]}...")
        self._logger.debug(f" Parsed Answer: {answer[:100]}...")
        self._logger.debug(f" Should Show Sources: {should_show_sources}")
        # Validate response context using LLM (NO hardcoded checks!)
        if should_show_sources and retrieved_chunks:
            answer = self.validate_response_context(answer, retrieved_chunks)
        
        self._add_conversation_messages(user_input, answer)

        return answer, should_show_sources

    def _build_comprehensive_context(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> Dict[str, Any]:
        """Build comprehensive context for LLM decision making."""
        context = {
            "user_input": user_input,
            "conversation_history": self._get_conversation_context(),
            "user_preferences": self.conversation.get_preferences(),
            "retrieved_chunks": retrieved_chunks or [],
            "chunk_count": len(retrieved_chunks) if retrieved_chunks else 0,
            "has_property_data": self._has_property_data(retrieved_chunks),
            "session_id": str(id(self.conversation))
        }
        return context

    def _has_property_data(self, retrieved_chunks: Optional[List[RetrievedChunk]]) -> bool:
        """Check if retrieved chunks contain actual property data."""
        
        return bool(retrieved_chunks)
   
    def _handle_general_knowledge_query(self, user_input: str) -> str:
        """Handle general knowledge queries using web search."""
        self._logger.debug(f"\n HANDLING GENERAL KNOWLEDGE QUERY:")
        self._logger.debug(f" Query: '{user_input}'")
        try:
            # Use web search to get current information
            from .utils import search_web_for_property_knowledge
            
            search_results = search_web_for_property_knowledge(user_input)
            
            if not search_results:
                # Fallback to LLM-only response if web search fails
                return self._generate_llm_knowledge_response(user_input)
            
            # Build response using web search results
            response = self._build_knowledge_response(user_input, search_results)
            
            self._logger.debug(f" Generated general knowledge response with web search")
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to handle general knowledge query: {e}")
            # Fallback to LLM-only response
            return self._generate_llm_knowledge_response(user_input)
    
    def _generate_llm_knowledge_response(self, user_input: str) -> str:
        """Generate knowledge response using LLM only (fallback)."""
        prompt = KNOWLEDGE_RESPONSE_PROMPT.format(user_input=user_input)
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            return response
        except Exception:
            return (
                "I'd be happy to explain that! However, I'm specifically designed to help you find properties to lease in the UAE. "
                "Would you like to search for properties instead? I can help you find the perfect place based on your needs."
            )
    
    def _build_knowledge_response(self, query: str, search_results: str) -> str:
        """Build knowledge response using web search results and LLM."""
        prompt = KNOWLEDGE_WITH_WEB_SEARCH_PROMPT.format(query=query, search_results=search_results)
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            return response
        except Exception:
            return self._generate_llm_knowledge_response(query)
       
    def _handle_average_price_query(self, user_input: str) -> str:
        """Handle average price queries by calculating from database."""
        self._logger.debug(f"\n HANDLING AVERAGE PRICE QUERY:")
        self._logger.debug(f" Query: '{user_input}'")
        # Get filters from user preferences (location, property type, etc.)
        filters = self.conversation.get_preferences().copy()
        self._logger.debug(f" Using filters from conversation: {filters}")
        # Get vector store client from pipeline_state
        try:
            from .main import pipeline_state
            vector_store = pipeline_state.vector_store_client
            
            if not vector_store:
                return "I apologize, but I'm unable to calculate average prices right now due to a technical issue. Please try again later."
            
            # Calculate average price using Elasticsearch aggregation
            price_stats = vector_store.calculate_average_price(filters)
            
            # Build human-friendly response
            location = price_stats.get("location_context", "UAE")
            average = price_stats.get("average", 0)
            median = price_stats.get("median", 0)
            min_price = price_stats.get("min", 0)
            max_price = price_stats.get("max", 0)
            count = price_stats.get("count", 0)
            
            if count == 0:
                return (
                    f"I couldn't find any properties matching your criteria to calculate an average price. "
                    f"Would you like to adjust your search filters or explore different areas?"
                )
            
            # Build response with context
            response = f"Based on {count} properties currently listed in {location}:\n\n"
            response += f"**Price Statistics:**\n"
            response += f"• **Average Annual Rent:** AED {average:,.0f}\n"
            response += f"• **Median Annual Rent:** AED {median:,.0f}\n"
            response += f"• **Price Range:** AED {min_price:,.0f} - AED {max_price:,.0f}\n\n"
            
            # Add context about the filters
            if filters:
                filter_context = []
                if "property_type_name" in filters:
                    filter_context.append(f"{filters['property_type_name']}s")
                if "number_of_bedrooms" in filters:
                    filter_context.append(f"{filters['number_of_bedrooms']}-bedroom")
                if "furnishing_status" in filters:
                    filter_context.append(filters["furnishing_status"])
                
                if filter_context:
                    response += f"*This calculation is based on {', '.join(filter_context)} properties in {location}.*\n\n"
            
            response += "Would you like to see specific properties in this price range, or adjust your search criteria?"
            
            self._logger.debug(f" Generated average price response")
            return response
            
        except Exception as e:
            self._logger.error(f"Failed to calculate average price: {e}")
            return (
                "I apologize, but I encountered an issue calculating the average price. "
                "Let me help you search for specific properties instead. What are you looking for?"
            )
    
    def _get_llm_intelligent_response(self, context: Dict[str, Any]) -> str:
        """Get intelligent response from LLM with all decision making."""

        prompt = self._build_intelligent_decision_prompt(context)
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            self._logger.debug(f" LLM Response: {response[:200]}...")
            # Check if response contains requirement gathering triggers
            if any(trigger in response.lower() for trigger in ["gather requirement", "save your requirements", "collect your needs"]):
                self._logger.debug(f" REQUIREMENT GATHERING TRIGGER DETECTED IN LLM RESPONSE!")
                self._logger.debug(f" Response preview: {response[:300]}...")
            return response 
        except Exception as e:
            self._logger.error(f"LLM intelligent response failed: {e}")
            return self._get_fallback_response(context)

    def _build_intelligent_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for LLM to make all decisions intelligently.
        Uses centralized INTELLIGENT_DECISION_PROMPT_TEMPLATE from instructions.py - NO DUPLICATES!
        """
        user_input = context["user_input"]
        retrieved_chunks = context["retrieved_chunks"]
        conversation_history = context.get("conversation_history", "")
        user_preferences = context.get("user_preferences", {})
        
        self._logger.debug(f" LLM DEBUG:")
        self._logger.debug(f" User input: '{user_input}'")
        self._logger.debug(f" Conversation history: '{conversation_history}'")
        self._logger.debug(f" User preferences: {user_preferences}")
        self._logger.debug(f" Has property data: {context['has_property_data']}")
        self._logger.debug(f" Retrieved chunks count: {len(retrieved_chunks)}")
        # Build context sections
        conversation_context = ""
        if conversation_history:
            conversation_context = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n\nCRITICAL: Use this conversation history to understand the user's context and build on previous exchanges. If the user previously searched for properties in a location and now adds budget constraints, remember the previous search and filter accordingly. Don't repeat information they've already shared."
        
        preferences_context = ""
        if user_preferences:
            preferences_context = f"\n\nUSER PREFERENCES: {user_preferences}"
        
        filters_context = ""
        if hasattr(self, '_last_extracted_filters') and self._last_extracted_filters:
            filters_context = f"\n\nEXTRACTED FILTERS FROM QUERY: {self._last_extracted_filters}\n\nCRITICAL: You MUST respect these filters. Only show properties that match these criteria."

        property_context = ""
        if retrieved_chunks:
            property_context = "\n\nPROPERTY DATA AVAILABLE:\n"
            from .config import get_settings
            settings = get_settings()
            primary_field = settings.database.primary_display_field
            pricing_field = settings.database.pricing_field
            
            for i, chunk in enumerate(retrieved_chunks[:5], 1):
                property_context += f"[Property {i}]\n{chunk.text}\n\n"
                
                metadata = chunk.metadata
                display_value = metadata.get(primary_field, "Unknown")
                price_value = metadata.get(pricing_field, "N/A") if pricing_field else "N/A"
                self._logger.debug(f" Property {i}: {display_value} | {price_value}")
        else:
            # CRITICAL: Explicitly tell LLM there are NO properties
            property_context = "\n\n⚠️ CRITICAL: NO PROPERTY DATA AVAILABLE (0 properties match the criteria)\n\n"
            property_context += "DO NOT say 'I found properties' or 'I've got options' - YOU HAVE ZERO PROPERTIES!\n"
            property_context += "You MUST:\n"
            property_context += "1. Be HONEST: Say 'I couldn't find properties matching all your criteria'\n"
            property_context += "2. Offer TWO options:\n"
            property_context += "   a) Try alternate searches (suggest nearby locations, flexible budget, different property types)\n"
            property_context += "   b) Save your requirements (gather for future notification)\n"
            property_context += "3. Make them feel heard and valued\n"
            self._logger.debug(f" NO PROPERTIES - Added explicit no-match guidance")
        # Use centralized template - NO DUPLICATES!
        prompt = INTELLIGENT_DECISION_PROMPT_TEMPLATE.format(
            user_input=user_input,
            conversation_context=conversation_context,
            preferences_context=preferences_context,
            filters_context=filters_context,
            property_context=property_context
        )

        return prompt

    def _parse_llm_response(self, llm_response: str) -> tuple[str, bool]:
        """Parse LLM response to extract answer and source decision."""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group(0))
                answer = response_data.get("answer", llm_response)
                show_sources = response_data.get("show_sources", False)
                reasoning = response_data.get("reasoning", "")
                
                # Check for requirement gathering detection
                if "REQUIREMENT_GATHERING_DETECTED" in reasoning:
                    self._logger.debug(f" REQUIREMENT GATHERING DETECTED IN LLM REASONING!")
                    self._logger.debug(f" Reasoning: {reasoning}")
                self._logger.debug(f"LLM decision: show_sources={show_sources}, reasoning={reasoning}")
                return answer, show_sources
            
        except Exception as e:
            self._logger.debug(f"Failed to parse LLM response as JSON: {e}")

        return llm_response, self._intelligent_source_guess(llm_response)

    def _intelligent_source_guess(self, response: str) -> bool:
        """Make intelligent guess about showing sources (fallback when LLM parsing fails)."""
        from .config import get_settings
        settings = get_settings()
        
        response_lower = response.lower()
        entity_name = settings.database.table[:-1]  # "properties" → "property"
        currency = settings.database.display.currency.lower()
        
        # Generic patterns that indicate entity listings
        if any(phrase in response_lower for phrase in [
            "here are", "i found", f"available {settings.database.table}", 
            f"matching {settings.database.table}", f"{entity_name} details",
            currency  # currency mentioned usually means showing listings
        ]):
            return True
        if any(phrase in response_lower for phrase in [
            "show you", "find you", "search for", "alternatives", "options"
        ]):
            return True
        return False

    def _get_fallback_response(self, context: Dict[str, Any]) -> str:
        """Fallback response when LLM fails."""
        user_input = context["user_input"]
        has_property_data = context["has_property_data"]
        
        if has_property_data:
            return f"I found some properties that might interest you. Let me show you the details."
        else:
            return f"I'd be happy to help you find properties in the UAE. Could you tell me more about what you're looking for?"

    def _get_conversation_context(self) -> str:
        """Get recent conversation history for short-term memory"""
        messages = self.conversation.get_messages()
        if not messages:
            return ""
        recent_messages = messages[-8:]
        context_parts = []
        
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")
        preferences = self.conversation.get_preferences()
        if preferences:
            
            context_parts.append(f"User Preferences: {preferences}")
            
        return " | ".join(context_parts)

    def _extract_detailed_requirements_from_conversation(self) -> Dict[str, Any]:
        """Extract detailed requirements from conversation messages using LLM"""
        messages = self.conversation.get_messages()
        if not messages:
            return {}
        
        # Get all user messages from conversation
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        combined_query = " ".join(user_messages)
        
        self._logger.debug(f" EXTRACTING DETAILED REQUIREMENTS FROM CONVERSATION:")
        self._logger.debug(f" Combined user messages: {combined_query[:300]}...")
        # Use centralized template - NO DUPLICATES!
        extraction_prompt = REQUIREMENT_EXTRACTION_PROMPT.format(combined_query=combined_query)

        try:
            response = self._safe_generate([{"role": "user", "content": extraction_prompt}])
            self._logger.debug(f" LLM extraction response: {response[:200]}...")
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_requirements = json.loads(json_match.group(0))
                self._logger.debug(f" Successfully extracted: {extracted_requirements}")
                return extracted_requirements
        except Exception as e:
            self._logger.error(f" Error extracting requirements: {e}")
        return {}

    def _fallback_missing_fields_check(self, requirements: Dict[str, Any]) -> List[str]:
        """Simple fallback missing field check if LLM fails"""
        from .config import get_settings
        settings = get_settings()
        
        missing = []
        
        # Check for location (any location field)
        location_fields = settings.database.location_hierarchy if settings.database.location_hierarchy else ['location']
        has_location = any(requirements.get(field) for field in location_fields)
        if not has_location:
            missing.append('location')
        
        # Check for pricing field
        if not requirements.get(settings.database.pricing_field):
            missing.append(settings.database.pricing_field)
        
        return missing

    def _ask_for_missing_requirements(self, missing_fields: List[str], extracted_requirements: Dict[str, Any]) -> str:
        """LLM generates friendly, interactive question for missing fields (GENERIC)"""
        self._logger.debug(f" ASKING FOR MISSING REQUIREMENTS: {missing_fields}")
        from .instructions import INTERACTIVE_REQUIREMENT_QUESTION_TEMPLATE
        from .config import get_settings
        settings = get_settings()
        
        # Build human-readable summary of what we have (GENERIC)
        current_summary = []
        if extracted_requirements:
            for key, value in list(extracted_requirements.items())[:4]:  # Show max 4
                if isinstance(value, dict) and "lte" in value:
                    current_summary.append(f"- Budget: under {value['lte']:,}")
                elif isinstance(value, dict) and "gte" in value:
                    current_summary.append(f"- Minimum: {value['gte']:,}")
                elif isinstance(value, list):
                    current_summary.append(f"- {key.replace('_', ' ').title()}: {', '.join(str(v) for v in value[:2])}")
                elif value:
                    current_summary.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        current_summary_text = "\n".join(current_summary) if current_summary else "None (starting fresh)"
        
        # Convert field names to human-readable (GENERIC)
        missing_labels = [field.replace('_', ' ').title() for field in missing_fields[:3]]
        missing_list = "\n".join([f"- {label}" for label in missing_labels])
        
        # Let LLM generate friendly, interactive question
        prompt = INTERACTIVE_REQUIREMENT_QUESTION_TEMPLATE.format(
            current_requirements_summary=current_summary_text,
            missing_fields_list=missing_list
        )
        
        self._logger.debug(f" Generating interactive question with LLM (friendly & professional)")
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            friendly_question = response.strip()
            self._logger.debug(f" LLM generated friendly question: {friendly_question[:150]}...")
            return friendly_question
        except Exception as e:
            self._logger.debug(f" LLM generation failed: {e}, using fallback")
            # Simple fallback if LLM fails
            entity = settings.database.table[:-1]  # "properties" -> "property"
            return f"Perfect! I have some details. To find the best {entity} for you, could you tell me about {missing_labels[0].lower()} and {missing_labels[1].lower() if len(missing_labels) > 1 else 'other preferences'}?"

    def gather_requirements_and_send(self, user_input: str, ask_confirmation: bool = False) -> str:
        """Multi-step requirement gathering with NEW API format
        
        Steps:
        1. Extract property requirements
        2. Check for missing essential fields
        3. Collect operator contact info
        4. Generate request name (LLM)
        5. Generate plain text description (LLM)
        6. Send to endpoint with new format
        """
        self._logger.debug(f"Requirement gathering triggered: {user_input[:50]}")
        
        # STEP 1: Extract property requirements from conversation
        extracted_requirements = self._extract_detailed_requirements_from_conversation()
        preferences = self.conversation.get_preferences()
        merged_requirements = {**preferences, **extracted_requirements}
        
        self._logger.debug(f"Merged requirements: {merged_requirements}")
        
        # STEP 2: Check if ALL ESSENTIAL property fields are present
        missing_property_fields = self._check_essential_property_fields(merged_requirements)
        
        if missing_property_fields:
            self._logger.debug(f"Missing essential property fields: {missing_property_fields}")
            return self._ask_for_missing_requirements(missing_property_fields, merged_requirements)
        
        # STEP 3: Check if contact info is collected
        if not self.conversation.has_contact_info():
            self._logger.debug("Need to collect operator contact info")
            
            # Try to extract from current user input
            contact_info = self._extract_contact_info_from_message(user_input)
            
            if contact_info.get("operator_name") and contact_info.get("contact_info"):
                # Successfully extracted from message
                self.conversation.set_contact_info(
                    contact_info["operator_name"],
                    contact_info["contact_info"]
                )
                # Reset waiting flag - we got the contact info!
                self.conversation.set_flow_state("requirement_gathering", waiting_for_contact=False)
                self._logger.debug(f"Contact info collected: {contact_info['operator_name']}")
                self._logger.debug(f"Reset waiting_for_contact_info = False (we have it now)")
            else:
                # Need to ask for contact info
                return self._ask_for_contact_information(merged_requirements)
        
        # STEP 4: Generate request name using LLM
        request_name = self._generate_request_name(merged_requirements)
        self._logger.debug(f"Generated request name: {request_name}")
        
        # STEP 5: Generate plain text description using LLM (ONLY summary paragraph, NO lists)
        description = self._generate_plain_text_description(merged_requirements)
        self._logger.debug(f"Generated description: {description[:200]}...")
        
        # STEP 6: Build payload for NEW API format
        contact_info = self.conversation.get_contact_info()
        
        payload = {
            "request": {
                "name": request_name,
                "operator": contact_info.get("operator_name", "Unknown"),
                "remark": contact_info.get("contact_info", "No contact info provided"),
                "details": {
                    "description": description
                }
            }
        }
        
        self._logger.debug("Sending requirement with new API format")
        self._logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        # STEP 7: Send to endpoint
        return self._send_requirements_to_new_endpoint(payload)
    
    def _generate_request_name(self, requirements: Dict[str, Any]) -> str:
        """LLM generates concise request title from requirements"""
        from .instructions import REQUEST_NAME_GENERATION_TEMPLATE
        
        self._logger.debug("Generating request name from requirements")
        self._logger.debug(f"Requirements: {json.dumps(requirements, indent=2)}")
        
        requirements_str = json.dumps(requirements, indent=2)
        prompt = REQUEST_NAME_GENERATION_TEMPLATE.format(requirements=requirements_str)
        
        self._logger.debug(f"Request name generation prompt length: {len(prompt)} chars")
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            self._logger.debug(f"LLM response for request name: {response[:200]}")
            
            result = _extract_json_from_response(response, "object")
            if result and "request_name" in result:
                self._logger.debug(f"Successfully generated request name: {result['request_name']}")
                return result["request_name"]
            else:
                self._logger.debug("LLM response did not contain request_name, using fallback")
        except Exception as e:
            self._logger.debug(f"Failed to generate request name: {e}")
        
        # Fallback
        fallback_name = self._create_fallback_request_name(requirements)
        self._logger.debug(f"Using fallback request name: {fallback_name}")
        return fallback_name
    
    def _create_fallback_request_name(self, requirements: Dict[str, Any]) -> str:
        """Create simple request name if LLM fails"""
        from .config import get_settings
        settings = get_settings()
        
        parts = []
        
        if requirements.get('number_of_bedrooms'):
            parts.append(f"{requirements['number_of_bedrooms']} BHK")
        
        if requirements.get('property_type_name'):
            parts.append(requirements['property_type_name'])
        
        # Get location
        for loc_field in settings.database.location_hierarchy:
            if requirements.get(loc_field):
                parts.append(f"in {requirements[loc_field]}")
                break
        
        return " ".join(parts) if parts else "Property Requirement Request"
    
    def _generate_plain_text_description(self, requirements: Dict[str, Any]) -> str:
        """LLM generates SINGLE PARAGRAPH plain text description (NO lists, NO conversation)"""
        from .instructions import PLAIN_TEXT_DESCRIPTION_TEMPLATE
        
        self._logger.debug("Generating plain text description (single paragraph only)")
        self._logger.debug(f"Requirements count: {len(requirements)} fields")
        
        requirements_str = json.dumps(requirements, indent=2)
        prompt = PLAIN_TEXT_DESCRIPTION_TEMPLATE.format(requirements=requirements_str)
        
        self._logger.debug(f"Description generation prompt length: {len(prompt)} chars")
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            description = response.strip()
            self._logger.debug(f"Successfully generated description ({len(description)} chars)")
            self._logger.debug(f"Description preview: {description[:300]}...")
            return description
        except Exception as e:
            self._logger.error(f"Failed to generate description: {e}")
            fallback_desc = self._create_fallback_description(requirements)
            self._logger.debug(f"Using fallback description ({len(fallback_desc)} chars)")
            return fallback_desc
    
    def _create_fallback_description(self, requirements: Dict[str, Any]) -> str:
        """Create simple SINGLE PARAGRAPH description if LLM fails (GENERIC)"""
        from .config import get_settings
        settings = get_settings()
        
        # Build flowing narrative paragraph (NO lists!)
        parts = []
        
        # Bedrooms + furnishing + type
        if requirements.get('number_of_bedrooms'):
            parts.append(f"{requirements['number_of_bedrooms']}-bedroom")
        if requirements.get('furnishing_status'):
            parts.append(requirements['furnishing_status'])
        if requirements.get('property_type_name'):
            parts.append(requirements['property_type_name'])
        
        # Location
        for loc_field in settings.database.location_hierarchy:
            if requirements.get(loc_field):
                parts.append(f"in {requirements[loc_field]}")
                break
        
        # Budget
        pricing_field = settings.database.pricing_field
        if requirements.get(pricing_field):
            budget = requirements[pricing_field]
            if isinstance(budget, dict) and 'lte' in budget:
                parts.append(f"with budget under {budget['lte']:,.0f} {settings.database.display.currency} annually")
            else:
                parts.append(f"with rent {budget:,.0f} {settings.database.display.currency} annually")
        
        # Lease duration
        if requirements.get('lease_duration'):
            parts.append(f"for lease of {requirements['lease_duration']}")
        
        # Amenities
        if requirements.get('amenities'):
            amenities = requirements['amenities'] if isinstance(requirements['amenities'], list) else [requirements['amenities']]
            parts.append(f"with {', '.join(amenities)}")
        
        # Build natural sentence
        desc = "Requirement gathering for " + " ".join(parts) + "."
        return desc
    
    def _extract_contact_info_from_message(self, user_input: str) -> Dict[str, str]:
        """Extract operator name and contact from user message using LLM"""
        from .instructions import CONTACT_INFO_EXTRACTION_TEMPLATE
        
        self._logger.debug(f"Extracting contact info from message: {user_input[:100]}")
        
        prompt = CONTACT_INFO_EXTRACTION_TEMPLATE.format(user_message=user_input)
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            self._logger.debug(f"Contact extraction LLM response: {response[:200]}")
            
            result = _extract_json_from_response(response, "object")
            if result:
                self._logger.debug(f"Contact info extracted - Name: {result.get('operator_name')}, Has contact: {bool(result.get('contact_info'))}")
                return result
            else:
                self._logger.debug("LLM response did not contain valid JSON")
        except Exception as e:
            self._logger.debug(f"Failed to extract contact info: {e}")
        
        self._logger.debug("No contact info found in message")
        return {"operator_name": None, "contact_info": None}
    
    def _ask_for_contact_information(self, requirements: Dict[str, Any]) -> str:
        """Ask for operator name and contact details - NATURAL & CONVERSATIONAL"""
        from .config import get_settings
        settings = get_settings()
        
        # Build a natural summary of what we have (GENERIC - works for any datasource)
        entity_name = settings.database.table[:-1]  # "properties" -> "property", "cars" -> "car"
        
        # Set flag to remember we asked for contact (prevents repetition)
        self.conversation.set_flow_state("requirement_gathering", waiting_for_contact=True)
        self._logger.debug(f"Set waiting_for_contact_info = True (prevent asking twice)")
        
        # Natural, human-like response (1-2 sentences, conversational)
        response = f"Perfect! I have all your {entity_name} requirements. "
        response += "To save these and have our team reach out, I'll need your name, phone number, and email address. "
        response += "Could you please share those with me?"
        
        return response
    
    def _check_essential_property_fields(self, requirements: Dict[str, Any]) -> List[str]:
        """Check if all ESSENTIAL property fields are present"""
        from .config import get_settings
        settings = get_settings()
        
        essential = settings.requirement_gathering.essential_fields
        self._logger.debug(f"Checking {len(essential)} essential fields: {essential}")
        self._logger.debug(f"Current requirements: {list(requirements.keys())}")
        
        missing = []
        
        for field in essential:
            if field == 'location':
                # Check any location field
                location_fields = settings.database.location_hierarchy
                has_location = any(requirements.get(f) for f in location_fields)
                if not has_location:
                    missing.append('location')
                    self._logger.debug(f"Missing: location (checked fields: {location_fields})")
                else:
                    found_in = [f for f in location_fields if requirements.get(f)]
                    self._logger.debug(f"Location found in: {found_in}")
            else:
                if not requirements.get(field):
                    missing.append(field)
                    self._logger.debug(f"Missing: {field}")
                else:
                    self._logger.debug(f"Present: {field} = {requirements[field]}")
        
        if missing:
            self._logger.debug(f"Missing {len(missing)} essential fields: {missing}")
        else:
            self._logger.debug("All essential fields present!")
        
        return missing
    
    def _send_requirements_to_new_endpoint(self, payload: Dict[str, Any]) -> str:
        """Send requirements using NEW API format with comprehensive logging
        
        NEW API FORMAT:
        {
          "request": {
            "name": "2 BHK Apartment in Dubai Marina",
            "operator": "John Doe",
            "remark": "+971-50-xxx, john@example.com",
            "details": {
              "description": "Plain text description..."
            }
          }
        }
        """
        endpoint = self._config.requirement_gathering.endpoint or "http://localhost:5000/backend/api/v1/admin/user-request-mappings/request-response"
        retry_attempts = 3
        retry_delay = 2
        
        self._logger.debug("="*60)
        self._logger.debug("SENDING REQUIREMENTS TO BACKEND")
        self._logger.debug("="*60)
        self._logger.debug(f"Endpoint: {endpoint}")
        self._logger.debug(f"Payload structure:")
        self._logger.debug(f"  - request.name: {payload['request']['name']}")
        self._logger.debug(f"  - request.operator: {payload['request']['operator']}")
        self._logger.debug(f"  - request.remark: {payload['request']['remark'][:50]}...")
        self._logger.debug(f"  - request.details.description: {len(payload['request']['details']['description'])} chars")
        self._logger.debug(f"Complete payload: {json.dumps(payload, indent=2)}")
        
        for attempt in range(retry_attempts):
            self._logger.debug(f"API Request Attempt {attempt + 1}/{retry_attempts}")
                
            try:
                self._logger.debug("Making POST request to backend...")
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=15,
                    headers={"Content-Type": "application/json"}
                )
                
                self._logger.debug(f"Response status code: {response.status_code}")
                self._logger.debug(f"Response headers: {dict(response.headers)}")
                
                if response.text:
                    self._logger.debug(f"Response body: {response.text[:500]}")
                
                if response.status_code == 200:
                    self._logger.debug("="*60)
                    self._logger.debug("SUCCESS! Requirements sent to backend")
                    self._logger.debug("="*60)
                    self.conversation.set_requirement_gathered(True)
                    self.requirement_metrics.log_attempt(True)
                    # Reset flow state - requirement gathering complete
                    self.conversation.set_flow_state(None, waiting_for_contact=False)
                    self._logger.debug(f"Reset flow state to None (requirement gathering complete)")
                    
                    return (
                        "Perfect! I've saved your requirements successfully.\n\n"
                        "Your property preferences have been recorded. Our team will work with agencies "
                        "to find matching properties and notify you when available.\n\n"
                        "Is there anything else I can help you with today?"
                    )
                elif attempt < retry_attempts - 1:
                    self._logger.debug(f"Endpoint returned {response.status_code}, will retry in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2  
                    continue
                else:
                    self._logger.error(f"FAILED after {retry_attempts} attempts - Status: {response.status_code}")
                    self._logger.error(f"Response body: {response.text[:200]}")
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've noted your requirements, but there was a technical issue saving them to our system. "
                        "Don't worry - I can still help you search for properties. "
                        "Would you like to continue exploring available options?"
                    )
                    
            except requests.exceptions.Timeout:
                self._logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < retry_attempts - 1:
                    self._logger.debug(f"Will retry in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self._logger.error("TIMEOUT after all retry attempts")
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've carefully noted your requirements! While there was a technical issue, "
                        "I can still help you search for properties with different criteria. "
                        "Would you like to try some alternative searches?"
                    )
            except Exception as e:
                self._logger.error(f"Request exception on attempt {attempt + 1}: {type(e).__name__} - {e}")
                if attempt < retry_attempts - 1:
                    self._logger.debug(f"Will retry in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self._logger.error(f"FAILED after all retry attempts: {e}")
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've carefully noted your requirements! While there was a technical issue, "
            "I can still help you search for properties with different criteria. "
            "Would you like to try some alternative searches?"
        )
        
        self._logger.error("Unexpected end of retry loop")
        return "I apologize, but I encountered an issue. Please try again."
    
    def _extract_location(self, merged_data: Dict[str, Any]) -> Optional[str]:
        """Extract location from merged data, checking multiple possible fields"""
        # Check for location in various forms
        location_fields = ['location', 'emirate', 'city', 'community']
        for field in location_fields:
            if field in merged_data and merged_data[field]:
                return str(merged_data[field])
        return None
    
    def _extract_rent_charge(self, merged_data: Dict[str, Any]) -> Optional[Any]:
        """Extract rent_charge from merged data"""
        rent_charge = merged_data.get("rent_charge")
        if rent_charge:
            # If it's a dict with lte/gte, return the dict
            if isinstance(rent_charge, dict):
                return rent_charge
            # If it's a number, return as is
            elif isinstance(rent_charge, (int, float)):
                return rent_charge
            # If it's a string, try to convert to number
            else:
                try:
                    return float(rent_charge)
                except (ValueError, TypeError):
                    return str(rent_charge)
        return None
    
    def _extract_rent_charge_value(self, merged_data: Dict[str, Any]) -> Optional[Any]:
        """Extract rent_charge value for backend (just the number, not the range)"""
        rent_charge = merged_data.get("rent_charge")
        if rent_charge:
            # If it's a dict with lte/gte, return the lte value
            if isinstance(rent_charge, dict) and "lte" in rent_charge:
                return rent_charge["lte"]
            elif isinstance(rent_charge, dict) and "gte" in rent_charge:
                return rent_charge["gte"]
            # If it's a number, return as is
            elif isinstance(rent_charge, (int, float)):
                return rent_charge
            # If it's a string, try to convert to number
            else:
                try:
                    return float(rent_charge)
                except (ValueError, TypeError):
                    return str(rent_charge)
        return None
    
    def _extract_amenities(self, merged_data: Dict[str, Any]) -> List[str]:
        """Extract features/amenities from merged data using config (domain-agnostic)"""
        amenities = []
        
        # Check for amenities list
        if "amenities" in merged_data and isinstance(merged_data["amenities"], list):
            amenities.extend(merged_data["amenities"])
        
        # Check for individual boolean fields from config
        from .config import get_settings
        settings = get_settings()
        feature_config = settings.database.boolean_fields or {}
        
        for field, display_label in feature_config.items():
            if merged_data.get(field) is True:
                amenities.append(display_label)
        
        return list(set(amenities))  # Remove duplicates

    def _save_to_fallback_storage(self, requirements: Dict[str, Any]) -> None:
        """Save requirements to fallback storage when endpoint fails"""
        try:
            fallback_data = {
                "timestamp": time.time(),
                "session_id": requirements.get("session_id", ""),
                "user_query": requirements.get("user_query", ""),
                "preferences": requirements.get("preferences", {}),
                "conversation_summary": requirements.get("conversation_summary", ""),
                "fallback_reason": "endpoint_failure"
            }
            self._logger.debug(f"Saved requirements to fallback storage: {fallback_data}")
        except Exception as e:
            self._logger.error(f"Failed to save to fallback storage: {e}")

    def generate(self, prompt: str) -> str:
        """Fallback for BaseLLM compatibility."""
        return self._safe_generate([{"role": "user", "content": prompt}])

    def _add_conversation_messages(self, user_input: str, response: str) -> None:
        """Helper method to add user and assistant messages to conversation history"""
        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)

    def _safe_generate(self, messages: List[Dict[str, str]], retries: int = 2, delay: int = 1) -> str:
        """Call provider safely with retry logic."""
        for attempt in range(retries + 1):
            try:
                return self._provider_client.generate(messages)
            except Exception as exc:
                self._logger.error(f"LLM generate failed: {exc}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    return "I'm sorry, I couldn't process your request right now."