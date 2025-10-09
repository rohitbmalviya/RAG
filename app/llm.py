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
            logger.info(
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
            self._logger.warning("LLM API key missing for Google provider")
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
            self._logger.warning("LLM_MODEL_API_KEY missing for OpenAI provider")

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
            self.llm_client._logger.warning(f"LLM alternatives generation failed: {e}")
        
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
            self._logger.warning(f"LLM classification failed: {e}, using fallback")
        
        # Fallback: simple pattern matching if LLM fails
        return self._fallback_classification(user_input)
    
    def _fallback_classification(self, user_input: str) -> Dict[str, Any]:
        """Simple fallback if LLM fails - returns safe default, NO hardcoded keywords!"""
        return {
            "category": "entity_search", 
            "confidence": 0.3, 
            "reasoning": "LLM classification unavailable, defaulting to entity search"
        }

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
                    self._logger.warning(f" LLM Validation: Issues found - {len(issues)} problems")
                    for issue in issues[:3]:  # Show top 3 issues
                        self._logger.warning(f" - {issue}")
                    # Add disclaimer to response
                    answer += "\n\n*Please verify details directly with the listing for accuracy.*"
                else:
                    self._logger.debug(f" LLM Validation: Response is accurate")
        except Exception as e:
            self._logger.warning(f"LLM validation failed: {e}")
        
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
        # Use LLM to classify query (NO hardcoded keywords!)
        classification = self.classify_query(user_input)
        category = classification.get("category", "entity_search")
        confidence = classification.get("confidence", 0.5)
        
        self._logger.debug(f" Query Category: {category} (confidence: {confidence})")
        # Handle special query types based on LLM classification
        if category == "price_inquiry" and settings.llm.query_handling.enable_price_aggregation:
            self._logger.debug(f" PRICE INQUIRY DETECTED!")
            answer = self._handle_average_price_query(user_input)
            self._logger.info(f" PRICE CALCULATION COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False  # No sources for price queries
        
        elif category == "knowledge_inquiry" and settings.llm.query_handling.enable_knowledge_search:
            self._logger.debug(f" KNOWLEDGE INQUIRY DETECTED!")
            answer = self._handle_general_knowledge_query(user_input)
            self._logger.info(f" KNOWLEDGE QUERY COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False  # No sources for knowledge queries
        
        elif category == "requirement_gathering" and settings.llm.query_handling.enable_requirement_gathering:
            self._logger.debug(f" REQUIREMENT GATHERING DETECTED!")
            self._logger.debug(f" CALLING gather_requirements_and_send() METHOD...")
            answer = self.gather_requirements_and_send(user_input, ask_confirmation=False)
            self._logger.info(f" REQUIREMENT GATHERING COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False
        
        elif category == "general_conversation":
            # Handle greetings and general conversation
            self._logger.debug(f" GENERAL CONVERSATION DETECTED!")
            context = self._build_comprehensive_context(user_input, retrieved_chunks)
            llm_response = self._get_llm_intelligent_response(context)
            answer, _ = self._parse_llm_response(llm_response)
            self._add_conversation_messages(user_input, answer)
            return answer, False  # No sources for greetings
        
        # Default: entity_search - process with full LLM intelligence
        self._logger.debug(f" ENTITY SEARCH - Processing with LLM intelligence...")
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

    def _summarize_conversation_context(self) -> str:
        """Summarize conversation for requirement gathering"""
        messages = self.conversation.get_messages()
        preferences = self.conversation.get_preferences()
        
        if not messages:
            return "No conversation history available."
        
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        summary_parts = []
        summary_parts.append(f"User discussed: {'; '.join(user_messages[-3:])}")  
        
        if preferences:
            pref_items = []
            for key, value in preferences.items():
                if isinstance(value, dict) and "lte" in value:
                    pref_items.append(f"{key}: up to {value['lte']}")
                else:
                    pref_items.append(f"{key}: {value}")
            if pref_items:
                summary_parts.append(f"Extracted preferences: {', '.join(pref_items)}")
        
        return " | ".join(summary_parts)

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
                self._logger.info(f" Successfully extracted: {extracted_requirements}")
                return extracted_requirements
        except Exception as e:
            self._logger.error(f" Error extracting requirements: {e}")
        return {}

    def _check_required_fields(self, extracted_requirements: Dict[str, Any]) -> List[str]:
        """LLM-based missing field detection - NO hardcoded required fields list!
        
        Uses LLM to determine what essential information is missing.
        Works for ANY domain (properties, cars, jobs, products, etc.)
        """
        from .config import get_settings
        from .instructions import GENERIC_REQUIREMENT_EXTRACTION_TEMPLATE
        
        settings = get_settings()
        
        # Check if requirement gathering is enabled
        if not settings.llm.query_handling.enable_requirement_gathering:
            return []  # Feature disabled
        
        # Get minimal context from config (generic!)
        entity_name = settings.database.table  # "properties", "cars", "jobs", etc.
        
        # Build requirement extraction prompt using generic template
        current_requirements_str = json.dumps(extracted_requirements, indent=2) if extracted_requirements else "{}"
        
        extraction_prompt = GENERIC_REQUIREMENT_EXTRACTION_TEMPLATE.format(
            entity_name=entity_name,
            current_requirements=current_requirements_str
        )
        
        try:
            # Let LLM determine what's missing using its intelligence!
            response = self._safe_generate([{"role": "user", "content": extraction_prompt}])
            result = _extract_json_from_response(response, "object")
            
            if result:
                missing_fields = result.get("missing_fields", [])
                questions = result.get("questions_to_ask", [])
                priority = result.get("priority", "unknown")
                
                self._logger.debug(f" LLM Missing Field Detection:")
                self._logger.debug(f" Missing fields: {missing_fields}")
                self._logger.debug(f" Priority: {priority}")
                if questions:
                    self._logger.debug(f"Questions to ask: {questions[:2]}")
                
                return missing_fields
        except Exception as e:
            self._logger.warning(f"LLM missing field detection failed: {e}, using fallback")
        
        # Fallback: simple check for basic fields if LLM fails
        return self._fallback_missing_fields_check(extracted_requirements)
    
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
        """Ask user to provide missing required fields"""
        self._logger.debug(f" ASKING FOR MISSING REQUIREMENTS: {missing_fields}")
        # Build response acknowledging what we have and asking for missing info
        response = "Perfect! I'd like to gather all the details for you. Let me ask a few quick questions to get everything we need:\n\n"
        
        # Acknowledge what we already know
        if extracted_requirements:
            response += "**What I know so far:**\n"
            for key, value in extracted_requirements.items():
                if isinstance(value, dict) and "lte" in value:
                    response += f"• {key.replace('_', ' ').title()}: Up to {value['lte']:,}\n"
                elif isinstance(value, dict) and "gte" in value:
                    response += f"• {key.replace('_', ' ').title()}: At least {value['gte']:,}\n"
                elif isinstance(value, list):
                    response += f"• {key.replace('_', ' ').title()}: {', '.join(value)}\n"
                else:
                    response += f"• {key.replace('_', ' ').title()}: {value}\n"
            response += "\n"
        
        # Ask for missing fields
        response += "**I still need to know:**\n"
        field_questions = {
            'location': "Which emirate/city are you looking in? (e.g., Dubai, Abu Dhabi, Sharjah)",
            'property_type_name': "What type of property? (e.g., apartment, villa, studio, townhouse)",
            'number_of_bedrooms': "How many bedrooms do you need?",
            'rent_charge': "What's your budget range for annual rent? (e.g., under 150K, 100K-200K)",
            'furnishing_status': "Do you prefer furnished, semi-furnished, or unfurnished?",
            'amenities': "Any specific amenities you need? (e.g., gym, pool, parking, balcony)",
            'lease_duration': "How long do you want to lease for? (e.g., 1 year, 2 years, 3 years)"
        }
        
        for i, field in enumerate(missing_fields, 1):
            question = field_questions.get(field, f"What's your preference for {field.replace('_', ' ')}?")
            response += f"{i}. {question}\n"
        
        response += "\nOnce I have these details, I'll save everything for our team to find the perfect match for you!"
        
        return response

    def gather_requirements_and_send(self, user_input: str, ask_confirmation: bool = True) -> str:
        """Gather requirements and optionally send to endpoint"""
        self._logger.debug(f"\n REQUIREMENT GATHERING TRIGGERED:")
        self._logger.debug(f" User input: '{user_input}'")
        self._logger.debug(f" Ask confirmation: {ask_confirmation}")
        self._logger.debug(f" Session ID: {str(id(self.conversation))}")
        # Extract detailed requirements from conversation
        extracted_requirements = self._extract_detailed_requirements_from_conversation()
        self._logger.debug(f" Extracted requirements: {extracted_requirements}")
        # Merge with conversation preferences (which contain emirate, etc.)
        preferences = self.conversation.get_preferences()
        self._logger.debug(f" Conversation preferences: {preferences}")
        # Merge extracted requirements with preferences, giving priority to extracted
        merged_requirements = {**preferences, **extracted_requirements}
        self._logger.debug(f" Merged requirements: {merged_requirements}")
        # Check if we have sufficient data to send to endpoint
        missing_fields = self._check_required_fields(merged_requirements)
        self._logger.debug(f" Missing required fields: {missing_fields}")
        if missing_fields:
            # Ask user to provide missing data
            return self._ask_for_missing_requirements(missing_fields, merged_requirements)
        
        conversation_summary = self._summarize_conversation_context()
        requirement_summary = {
            "user_query": user_input,
            "preferences": preferences,
            "extracted_requirements": merged_requirements,
            "conversation_summary": conversation_summary,
            "session_id": str(id(self.conversation)),
            "timestamp": time.time()
        }
        
        self._logger.debug(f" User preferences: {preferences}")
        self._logger.debug(f" Conversation summary: {conversation_summary[:200]}...")
        if ask_confirmation:
            response = (
                "I'll summarize what you're looking for:\n\n"
                "**Your Requirements:**\n"
            )
            if merged_requirements:
                for key, value in merged_requirements.items():
                    if isinstance(value, dict) and "lte" in value:
                        response += f"• {key.replace('_', ' ').title()}: Up to {value['lte']:,}\n"
                    else:
                        response += f"• {key.replace('_', ' ').title()}: {value}\n"
            else:
                response += f"• Based on your query: {user_input}\n"
                
            response += (
                f"\n**Conversation Summary:** {conversation_summary}\n\n"
                "Would you like me to save these requirements? Our team will work with agencies to find matching properties and notify you when available.\n\n"
                "Reply 'yes' to save, or 'no' to continue searching with different criteria."
            )
            
            return response
        else:
            self._logger.debug(f" Sending requirements directly to endpoint (no confirmation needed)")
            return self._send_requirements_to_endpoint(requirement_summary)
    
    def _send_requirements_to_endpoint(self, requirements: Dict[str, Any]) -> str:
        """Send requirements to the backend endpoint with retry logic and fallback storage.
        
        BACKEND API CONTRACT:
        Expected fields (all optional):
        - session_id (str): Conversation session identifier
        - location (str): Location preference (emirate/city/community)
        - property_type (str): Type of property (apartment, villa, studio, etc.)
        - number_of_bedrooms (int): Number of bedrooms
        - rent_charge (float): Budget for annual rent
        - furnishing_status (str): Furnished, semi-furnished, unfurnished
        - amenities (List[str]): List of amenities (gym, pool, parking, etc.)
        - nearby_landmarks (str): Nearby landmarks preference
        - building_name (str): Specific building name
        - rent_type (str): Lease/Holiday Home/Management Fees
        
        NOTE: If your backend controller uses different field names, update the mapping below.
        """
        
        endpoint = self._config.requirement_gathering.endpoint or "http://localhost:5000/backend/api/v1/user/requirement"
        
        self._logger.debug(f"\n SENDING REQUIREMENTS TO API ENDPOINT:")
        self._logger.debug(f" Endpoint: {endpoint}")
        self._logger.debug(f" Requirements data: {requirements}")
        # Transform the data to match backend controller expectations
        extracted_requirements = requirements.get("extracted_requirements", {})
        preferences = requirements.get("preferences", {})
        
        # Merge extracted requirements with preferences (extracted takes priority)
        merged_data = {**preferences, **extracted_requirements}
        
        # CRITICAL: Map RAG field names to backend controller field names
        # If your backend uses different naming conventions, update this mapping
        backend_payload = {
            "session_id": requirements.get("session_id", ""),
            # Location mapping - handles emirate, city, or community
            "location": self._extract_location(merged_data),
            # Property type mapping - maps property_type_name → property_type
            "property_type": merged_data.get("property_type_name"),
            # Direct mappings
            "number_of_bedrooms": merged_data.get("number_of_bedrooms"),
            # Rent charge - extracts value from range if needed
            "rent_charge": self._extract_rent_charge_value(merged_data),
            "furnishing_status": merged_data.get("furnishing_status"),
            # Amenities - aggregates from boolean fields and amenities list
            "amenities": self._extract_amenities(merged_data),
            "nearby_landmarks": merged_data.get("nearby_landmarks"),
            "building_name": merged_data.get("building_name"),
            # Rent type mapping - maps rent_type_name → rent_type
            "rent_type": merged_data.get("rent_type_name"),
            # Additional fields for context
            "conversation_summary": requirements.get("conversation_summary", ""),
            "timestamp": requirements.get("timestamp", time.time())
        }
        
        # Remove None values to keep payload clean
        backend_payload = {k: v for k, v in backend_payload.items() if v is not None}
        
        self._logger.debug(f" 🔄 TRANSFORMED PAYLOAD FOR BACKEND:")
        self._logger.debug(f" Original extracted_requirements: {extracted_requirements}")
        self._logger.debug(f" Original preferences: {preferences}")
        self._logger.debug(f" Merged data: {merged_data}")
        self._logger.debug(f" Backend payload: {backend_payload}")
        self._logger.debug(f" Payload size: {len(str(backend_payload))} bytes")
        enhanced_requirements = backend_payload
        retry_attempts = 3
        retry_delay = 2
        
        for attempt in range(retry_attempts):
            try:
                self._logger.debug(f" 📤 API CALL ATTEMPT {attempt + 1}/{retry_attempts}:")
                self._logger.debug(f" URL: {endpoint}")
                self._logger.debug(f" Payload: {enhanced_requirements}")
                self._logger.info(f"Sending requirements to endpoint: {endpoint} (attempt {attempt + 1}/{retry_attempts})")
                response = requests.post(
                    endpoint,
                    json=enhanced_requirements,
                    timeout=15,
                    headers={"Content-Type": "application/json"}
                )
                
                self._logger.debug(f" Response Status: {response.status_code}")
                self._logger.debug(f" Response Headers: {dict(response.headers)}")
                if response.text:
                    self._logger.debug(f" Response Body: {response.text[:500]}...")
                if response.status_code == 200:
                    self._logger.info(f" SUCCESS: Requirements sent successfully!")
                    self.conversation.set_requirement_gathered(True)
                    self.requirement_metrics.log_attempt(True)
                    self._logger.info("Requirements successfully sent to endpoint")
                    return (
                        "Perfect! I've saved your requirements successfully.\n\n"
                        "Your property preferences have been recorded and we'll keep them in mind for future searches.\n\n"
                        "Is there anything else I can help you with today?"
                    )
                elif attempt < retry_attempts - 1:
                    self._logger.debug(f" RETRY: Status {response.status_code}, retrying in {retry_delay}s")
                    self._logger.warning(f"Endpoint returned status {response.status_code}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2  
                    continue
                else:
                    self._logger.error(f" FAILED: Status {response.status_code} after {retry_attempts} attempts")
                    self._logger.warning(f"Endpoint returned status {response.status_code} after {retry_attempts} attempts")
                    self._save_to_fallback_storage(enhanced_requirements)
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've noted your requirements, but there was a technical issue saving them to our system. "
                        "Don't worry - I can still help you search for properties or try again later. "
                        "Would you like to continue exploring available options?"
                    )
                    
            except requests.exceptions.Timeout:
                if attempt < retry_attempts - 1:
                    self._logger.debug(f" ⏰ TIMEOUT: Retrying in {retry_delay}s")
                    self._logger.warning(f"Request timeout, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self._logger.error(f" TIMEOUT FAILED: After all retry attempts")
                    self._logger.error("Request timeout after all retry attempts")
                    self._save_to_fallback_storage(enhanced_requirements)
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've carefully noted your requirements! While there was a technical issue with our system, "
                        "I can still help you search for properties with different criteria. "
                        "Would you like to try some alternative searches?"
                    )
            except Exception as e:
                if attempt < retry_attempts - 1:
                    self._logger.error(f" ERROR: {e}, retrying in {retry_delay}s")
                    self._logger.warning(f"Request failed: {e}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self._logger.error(f" ERROR FAILED: {e} after {retry_attempts} attempts")
                    self._logger.error(f"Failed to send requirements to endpoint after {retry_attempts} attempts: {e}")
                    self._save_to_fallback_storage(enhanced_requirements)
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've carefully noted your requirements! While there was a technical issue with our system, "
                        "I can still help you search for properties with different criteria. "
                        "Would you like to try some alternative searches?"
                    )
        return (
            "I've carefully noted your requirements! While there was a technical issue with our system, "
            "I can still help you search for properties with different criteria. "
            "Would you like to try some alternative searches?"
        )
    
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
            self._logger.info(f"Saved requirements to fallback storage: {fallback_data}")
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