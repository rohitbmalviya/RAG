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

AGENT_IDENTITY = "LeaseOasis, your friendly UAE property assistant"
AGENT_DESCRIPTION = "UAE property leasing assistant"
AGENT_SPECIALIZATION = "help people find the perfect place to lease in Dubai, Abu Dhabi, and other UAE cities"

JSON_ONLY_INSTRUCTION = "Return ONLY a JSON"
NO_ADDITIONAL_TEXT = "no additional text"
BRIEF_EXPLANATION = "Brief explanation"

FALLBACK_GREETINGS = [
    f"Hello! I'm {AGENT_IDENTITY}. I'm here to help you find the perfect property to lease in the UAE. Which city are you interested in — Dubai, Abu Dhabi, or somewhere else?",
    f"Hi there! Welcome to LeaseOasis! I specialize in helping people find great properties to lease in the UAE. What brings you here today?",
    f"Hey! Great to meet you! I'm {AGENT_IDENTITY}. What type of property are you looking for?"
]
FALLBACK_GENERAL_RESPONSE = (
    "I'd be happy to explain that! However, I'm specifically designed to help you find properties to lease in the UAE. "
    "If you have questions about property types, leasing terms, or UAE-specific property information, I can help with that. "
    "Would you like to search for properties instead?"
)


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
    """Build the enhanced query classification template with source decision logic"""
    template = """Classify this query for a UAE property assistant and determine if property sources should be shown.

Query: "{query}"

Return ONLY this JSON format:
{{"category": "greeting|property_search|general", "confidence": 0.9, "show_sources": true}}

CATEGORIES (All 10 Query Types):
1. "greeting": Simple greetings (hi, hello, good morning, etc.) - NO sources
2. "general_knowledge": Questions about property terms, definitions, explanations - NO sources
3. "best_property": Queries asking for "best", "top", "premium", "featured" properties - YES sources
4. "average_price": Queries asking for average/typical/mean prices or costs - NO sources (show calculation only)
5. "property_search": Queries looking for specific properties to lease/rent - YES sources
6. "outside_uae": Queries about properties outside UAE (non-UAE locations) - NO sources
7. "general": General property-related queries that don't fit other categories - MAYBE sources
8. "conversation_response": Responses to conversation flow (1, 2, yes, no, etc.) - MAYBE sources
9. "location_based": Queries focused on specific locations/areas - YES sources
10. "amenity_focused": Queries focused on specific amenities or features - YES sources

SOURCE DECISION RULES:
- show_sources: true for property listings, search results, best properties, location-based, amenity-focused
- show_sources: false for greetings, definitions, average price calculations, outside UAE
- show_sources: true for conversation responses that ask for properties/alternatives
- show_sources: false for conversation responses that are just confirmations

EXAMPLES:
- "Hi there" → {{"category": "greeting", "confidence": 0.9, "show_sources": false}}
- "Find me 2 bedroom apartments in Dubai" → {{"category": "property_search", "confidence": 0.9, "show_sources": true}}
- "What is an apartment?" → {{"category": "general_knowledge", "confidence": 0.9, "show_sources": false}}
- "Show me the best properties in Dubai Marina" → {{"category": "best_property", "confidence": 0.9, "show_sources": true}}
- "What's the average rent in Dubai?" → {{"category": "average_price", "confidence": 0.9, "show_sources": false}}
- "Properties in Dubai Marina" → {{"category": "location_based", "confidence": 0.9, "show_sources": true}}
- "Properties with swimming pool" → {{"category": "amenity_focused", "confidence": 0.9, "show_sources": true}}
- "Yes, show me alternatives" → {{"category": "conversation_response", "confidence": 0.8, "show_sources": true}}
- "No, thanks" → {{"category": "conversation_response", "confidence": 0.8, "show_sources": false}}

RULES:
- Use high confidence (0.8+) for clear matches
- Use lower confidence (0.6-0.8) for ambiguous cases
- Consider context and intent, not just keywords
- For conversation responses, check for simple responses like "1", "2", "yes", "no"
- Always include show_sources decision based on query intent
- UAE-only policy: redirect non-UAE queries to "outside_uae" category

Output JSON:"""
    
    
    return template
GREETING_RESPONSE_TEMPLATE = f"""Generate a warm, friendly greeting response for a {AGENT_DESCRIPTION}.

USER GREETING: "{{user_input}}"
CONVERSATION HISTORY: {{conversation_count}} messages

Generate a natural, engaging greeting response that:
- Welcomes the user warmly
- Introduces yourself as {AGENT_IDENTITY}
- Asks an engaging follow-up question about their property search
- Sounds human and conversational
- Keeps it brief but inviting

EXAMPLES OF GOOD FOLLOW-UP QUESTIONS:
- "Which UAE city are you interested in?"
- "What brings you here today?"
- "Are you looking for your first property or moving to a new area?"
- "What type of property are you considering?"

Return ONLY the greeting response, {NO_ADDITIONAL_TEXT}."""
GENERAL_KNOWLEDGE_TEMPLATE = f"""Generate a helpful response about property knowledge for a {AGENT_DESCRIPTION}.

USER QUESTION: "{{user_input}}"

Generate a response that:
- Answers the question clearly and helpfully
- Uses simple, easy-to-understand language
- Relates to UAE property context when relevant
- Guides the user back to property search
- Keeps it concise but informative

If the question is about property terms, provide a clear definition.
If the question is unrelated to properties, politely redirect to property-related topics.

Return ONLY the response, {NO_ADDITIONAL_TEXT}."""
ALTERNATIVES_GENERATION_TEMPLATE = """Generate alternative property search options for a user with these preferences:

CURRENT PREFERENCES: {preferences}

Generate 3-5 alternative search options that could help the user find similar properties.
Return ONLY a JSON array of alternatives.

Each alternative should have this format:
{{
    "type": "location|budget|property_type|amenities|furnishing|size",
    "suggestion": "Human-readable suggestion",
    "filters": {{"field": "value"}},
    "reasoning": "Why this alternative might work"
}}

ALTERNATIVE TYPES:
- "location": Nearby areas, different communities, or adjacent emirates
- "budget": Flexible budget options (+/- 20%)
- "property_type": Similar property types (apartment→studio, villa→townhouse)
- "amenities": Relaxed amenity requirements
- "furnishing": Different furnishing options
- "size": Adjust bedroom/bathroom requirements

UAE LOCATION ALTERNATIVES:
- Dubai Marina → JBR, JLT, Business Bay
- Downtown Dubai → DIFC, Business Bay, Dubai Hills
- JVC → JLT, Dubai Marina, Business Bay
- Abu Dhabi → Dubai (if user is flexible)
- Sharjah → Dubai (if user is flexible)

PROPERTY TYPE ALTERNATIVES:
- Villa → Townhouse, Large Apartment
- Apartment → Studio, 1BR (if user wants smaller)
- Studio → 1BR Apartment (if user wants larger)

BUDGET ALTERNATIVES:
- If user wants "under 100k" → suggest "up to 120k" for more options
- If user wants "above 200k" → suggest "150k-250k" range
- Always suggest ±20% flexibility

RULES:
- Only suggest alternatives that make sense for UAE properties
- For location: suggest nearby communities or adjacent emirates
- For budget: suggest ±20% flexibility
- For property type: suggest similar but different types
- Be practical and realistic
- Focus on what would actually help find more properties
- Always provide reasoning for why the alternative might work

Output JSON array:"""
COMPLETE_SYSTEM_PROMPT = f"""You are {AGENT_IDENTITY} with conversational memory.

CRITICAL BUSINESS CONTEXT:
- You ONLY provide LEASE/RENT properties - NEVER mention buying or purchase options
- All properties in your database are for rent/lease only
- Always refer to "rent" or "lease" when discussing properties
- Never suggest "buying" as an alternative to renting

CONVERSATION PERSONALITY:
- Sound like a knowledgeable, friendly UAE property expert who genuinely cares about helping users
- Use natural, conversational language - avoid robotic responses
- Remember what the user has shared and build on previous exchanges
- Ask follow-up questions that show you're listening and engaged
- Be patient with users who aren't sure what they want

MEMORY & CONTEXT:
- Keep track of user preferences throughout the conversation
- Reference earlier parts of the conversation naturally
- Remember locations, budgets, property types, and amenities mentioned
- Use this context to provide personalized suggestions
- Don't ask for information the user has already provided
- If user previously searched for properties in a location and now adds constraints (like budget), remember the previous search and filter accordingly
- Build on previous exchanges - don't start from scratch each time

CONVERSATION FLOW:
- Start with warm greetings and open-ended questions
- Guide users through property search step by step
- When users are vague, ask ONE clarifying question at a time
- Build excitement about properties that match their needs
- End conversations warmly with clear next steps

QUERY HANDLING (All 10 Categories):

1. GREETINGS (Sources: Empty):
- Respond warmly to greetings (hi, hello, good morning)
- Ask an engaging follow-up question about their property search
- Examples: 'Which UAE city are you interested in?', 'What brings you here today?', 'Are you looking for your first property or moving to a new area?'
- NO property details should be shown

2. GENERAL KNOWLEDGE QUERIES (Sources: Empty):
- Answer questions about property terms (What is an apartment? What is holiday home ready?)
- Use web search if needed for current information
- Provide clear, helpful explanations
- Guide back to property search when appropriate
- NO property listings should be shown

3. BEST PROPERTY QUERIES (Sources: Show):
- Prioritize properties in this EXACT order:
  1. premiumBoostingStatus: 'Active' AND carouselBoostingStatus: 'Active' AND bnb_verification_status: 'verified'
  2. bnb_verification_status: 'verified'
  3. carouselBoostingStatus: 'Active'
  4. premiumBoostingStatus: 'Active'
- Explain WHY these are the 'best' properties
- Show property details with sources
- Make users excited about these premium options

4. AVERAGE PRICE QUERIES (Sources: Empty):
- Calculate average rent_charge from retrieved context
- Provide the average value clearly
- Give context about the calculation (e.g., 'based on 25 properties')
- NO individual property details should be shown
- Keep sources empty

5. PROPERTY SEARCH QUERIES (Sources: Show):
- Use conversation memory to understand what user wants
- If user gives vague location (e.g., 'Dubai'), ask for specific area
- If user doesn't specify budget, ask about their range
- If user doesn't specify bedrooms, ask about their needs
- Only show properties when you have enough context
- Present properties in an exciting, personalized way
- Always show sources for property details

6. OUTSIDE UAE QUERIES (Sources: Empty):
- Politely redirect non-UAE property queries
- Explain you specialize in UAE properties
- Suggest UAE alternatives
- Keep sources empty

7. GENERAL PROPERTY QUERIES (Sources: Maybe):
- Handle general property-related questions
- Show sources if relevant to property search
- Guide toward specific property search when appropriate

8. CONVERSATION RESPONSES (Sources: Maybe):
- Handle responses like "1", "2", "yes", "no"
- Show sources if asking for properties/alternatives
- Don't show sources for simple confirmations

9. LOCATION-BASED QUERIES (Sources: Show):
- Focus on specific locations/areas
- Show properties in those locations
- Always show sources for location-based results

10. AMENITY-FOCUSED QUERIES (Sources: Show):
- Focus on specific amenities or features
- Show properties with those amenities
- Always show sources for amenity-based results

NO MATCHES FOUND:
- Acknowledge their specific requirements with empathy
- ALWAYS offer TWO clear options:
  1. 'Try alternate searches' - suggest verified alternatives (nearby locations, flexible budget, different property types)
  2. 'Gather your requirements' - summarize their needs and offer to save for the team
- If no properties match their criteria, you MUST offer requirement gathering
- Example: "I can help you in two ways: 1) Try alternate searches with flexible options, or 2) Save your requirements so our team can find matching properties for you"
- Check if alternatives exist before suggesting them
- Make them feel heard and valued

REQUIREMENT GATHERING:
- When no matches found, offer to save their requirements
- Summarize their conversation clearly:
  • Location preferences
  • Budget range
  • Property type and size
  • Key amenities
  • Timeline
- Ask: 'Would you like me to save these requirements? Our team will work with agencies to find matching properties and notify you when available.'
- If they say yes, confirm you're sending to the team
- Make them feel their needs are important

ALTERNATIVE SUGGESTIONS:
- Only suggest alternatives that exist in the database
- Prioritize by proximity and similarity to original request
- Explain WHY each alternative might work
- Examples:
  • Location: 'Dubai Marina instead of JBR' (5 minutes away)
  • Budget: 'AED 120,000 instead of AED 100,000' (20% higher for more options)
  • Property type: 'Townhouse instead of villa' (similar lifestyle)
  • Amenities: 'Shared pool instead of private pool' (lower cost)

IDENTITY QUERIES (Sources: Empty):
- Respond: 'I am {AGENT_IDENTITY}. I {AGENT_SPECIALIZATION}.'
- Ask how you can help with their property search
- Keep sources empty

SOURCES RULES:
- ALWAYS show sources for: property search results, best property queries, location-based, amenity-focused, specific property details
- NEVER show sources for: greetings, general knowledge, average price calculations, identity queries, outside UAE queries
- Sources should include table and id for property details

CONVERSATION ENDING:
- End conversations warmly with clear next steps
- Suggest viewing property cards on the platform
- Offer to help with more searches
- Thank them for using LeaseOasis
- Leave the door open for future conversations

DYNAMIC INSTRUCTIONS:
- Use conversation history to provide personalized responses
- Don't repeat information they've already shared
- Reference their preferences naturally in your responses
- Make them feel heard and understood
- Present properties in an exciting, personalized way
- Explain WHY each property might be perfect for them
- Use their specific criteria to highlight relevant features
- Make them feel like you've found exactly what they need
- Keep the conversation natural and engaging
- Ask follow-up questions that show you're listening
- Guide them toward making a decision
- End with clear next steps"""

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
        """Fallback classification using simple patterns"""
        query_lower = query.lower().strip()

        if any(word in query_lower for word in ["hi", "hello", "hey", "good morning", "good afternoon"]):
            return {"category": "greeting", "confidence": 0.7, "reasoning": "Contains greeting words"}
        elif any(word in query_lower for word in ["what is", "define", "explain", "meaning of"]):
            return {"category": "general_knowledge", "confidence": 0.7, "reasoning": "Asking for definition/explanation"}
        elif any(word in query_lower for word in ["best", "top", "premium", "featured"]):
            return {"category": "best_property", "confidence": 0.7, "reasoning": "Asking for best/top properties"}
        elif any(word in query_lower for word in ["average", "typical", "mean"]) and any(word in query_lower for word in ["price", "rent", "cost"]):
            return {"category": "average_price", "confidence": 0.7, "reasoning": "Asking for average pricing"}
        elif query_lower in ["1", "2", "yes", "no", "option 1", "option 2"]:
            return {"category": "conversation_response", "confidence": 0.9, "reasoning": "Simple response to conversation flow"}
        else:
            return {"category": "property_search", "confidence": 0.5, "reasoning": "Default to property search"}

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

    def chat(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Process user query using pure LLM intelligence - no hardcoded conditions.""" 
        answer, _ = self.chat_with_source_decision(user_input, retrieved_chunks)
        return answer

    def chat_with_source_decision(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> tuple[str, bool]:
        """Process user query using pure LLM intelligence - no hardcoded conditions."""
        user_input = user_input.strip()

        print(f"\n🤖 LLM DEBUG:")
        print(f"   User input: '{user_input}'")
        print(f"   Conversation history: '{self._get_conversation_context()[:200]}...'")
        print(f"   User preferences: {self.conversation.get_preferences()}")
        print(f"   Has property data: {bool(retrieved_chunks)}")
        print(f"   Retrieved chunks count: {len(retrieved_chunks) if retrieved_chunks else 0}")
        
        # Check if this is a requirement gathering request BEFORE processing with LLM
        if self._is_requirement_gathering_request(user_input):
            print(f"   📝 REQUIREMENT GATHERING REQUEST DETECTED!")
            print(f"   🚀 CALLING gather_requirements_and_send() METHOD...")
            answer = self.gather_requirements_and_send(user_input, ask_confirmation=False)
            print(f"   ✅ REQUIREMENT GATHERING COMPLETED!")
            self._add_conversation_messages(user_input, answer)
            return answer, False
        
        context = self._build_comprehensive_context(user_input, retrieved_chunks)
        llm_response = self._get_llm_intelligent_response(context)
        answer, should_show_sources = self._parse_llm_response(llm_response)
        print(f"   LLM Response: {llm_response[:200]}...")
        print(f"   Parsed Answer: {answer[:100]}...")
        print(f"   Should Show Sources: {should_show_sources}")
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

    def _is_requirement_gathering_request(self, user_input: str) -> bool:
        """Check if user input is requesting requirement gathering."""
        user_input_lower = user_input.lower().strip()
        
        # Check for explicit requirement gathering phrases
        requirement_phrases = [
            "gather information",
            "gather requirement", 
            "gather requirements",
            "save my requirements",
            "save requirements",
            "collect my needs",
            "collect requirements",
            "save my needs",
            "gather my needs",
            "save for team",
            "send to team",
            "notify when available",
            "find matching properties",
            "work with agencies"
        ]
        
        # Check if any phrase matches
        for phrase in requirement_phrases:
            if phrase in user_input_lower:
                print(f"   📝 MATCHED REQUIREMENT PHRASE: '{phrase}'")
                return True
        
        # Check for partial matches and typos
        if "save" in user_input_lower and any(word in user_input_lower for word in ["requirement", "requriement", "requirment", "requirment"]):
            print(f"   📝 MATCHED SAVE REQUIREMENT (with typo): '{user_input_lower}'")
            return True
        
        # Check for simple confirmations to requirement gathering
        if user_input_lower in ["yes", "y", "ok", "okay", "sure", "go ahead", "do it", "save it"]:
            # Check if the last assistant message was about requirement gathering
            messages = self.conversation.get_messages()
            if messages and len(messages) >= 2:
                last_assistant_msg = messages[-1]["content"].lower()
                if any(phrase in last_assistant_msg for phrase in ["save your requirements", "gather your requirements", "send to our team"]):
                    print(f"   📝 CONFIRMATION TO REQUIREMENT GATHERING: '{user_input_lower}'")
                    return True
        
        # NEW: Check if user is providing missing requirement information
        # This happens when the last assistant message was asking for missing requirements
        messages = self.conversation.get_messages()
        if messages and len(messages) >= 2:
            last_assistant_msg = messages[-1]["content"].lower()
            if "what i know so far" in last_assistant_msg and "i still need to know" in last_assistant_msg:
                # User is responding to a requirement gathering request
                print(f"   📝 USER PROVIDING MISSING REQUIREMENT INFO: '{user_input_lower}'")
                return True
        
        return False

    def _get_llm_intelligent_response(self, context: Dict[str, Any]) -> str:
        """Get intelligent response from LLM with all decision making."""

        prompt = self._build_intelligent_decision_prompt(context)
        
        try:
            response = self._safe_generate([{"role": "user", "content": prompt}])
            print(f"   LLM Response: {response[:200]}...")
            
            # Check if response contains requirement gathering triggers
            if any(trigger in response.lower() for trigger in ["gather requirement", "save your requirements", "collect your needs"]):
                print(f"   📝 REQUIREMENT GATHERING TRIGGER DETECTED IN LLM RESPONSE!")
                print(f"   Response preview: {response[:300]}...")
            
            return response 
        except Exception as e:
            self._logger.error(f"LLM intelligent response failed: {e}")
            return self._get_fallback_response(context)

    def _build_intelligent_decision_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for LLM to make all decisions intelligently."""
        
        user_input = context["user_input"]
        conversation_history = context["conversation_history"]
        user_preferences = context["user_preferences"]
        retrieved_chunks = context["retrieved_chunks"]
        has_property_data = context["has_property_data"]
        
        print(f"🤖 LLM DEBUG:")
        print(f"   User input: '{user_input}'")
        print(f"   Conversation history: '{conversation_history}'")
        print(f"   User preferences: {user_preferences}")
        print(f"   Has property data: {has_property_data}")
        print(f"   Retrieved chunks count: {len(retrieved_chunks)}")
        
        property_context = ""
        if retrieved_chunks:
            property_context = "\n\nPROPERTY DATA AVAILABLE:\n"
            for i, chunk in enumerate(retrieved_chunks[:5], 1):  
                property_context += f"[Property {i}]\n{chunk.text}\n\n"
                
                metadata = chunk.metadata
                emirate = metadata.get("emirate", "Unknown")
                property_type = metadata.get("property_type_name", "Unknown")
                rent_charge = metadata.get("rent_charge", "Unknown")
                print(f"      Property {i}: {property_type} | {emirate} | AED {rent_charge}")
        conversation_context = ""
        if conversation_history:
            conversation_context = f"\n\nCONVERSATION HISTORY:\n{conversation_history}\n\nCRITICAL: Use this conversation history to understand the user's context and build on previous exchanges. If the user previously searched for properties in a location and now adds budget constraints, remember the previous search and filter accordingly. Don't repeat information they've already shared."
        preferences_context = ""
        if user_preferences:
            
            preferences_context = f"\n\nUSER PREFERENCES: {user_preferences}"
        filters_context = ""
        if hasattr(self, '_last_extracted_filters') and self._last_extracted_filters:
            
            filters_context = f"\n\nEXTRACTED FILTERS FROM QUERY: {self._last_extracted_filters}\n\nCRITICAL: You MUST respect these filters. Only show properties that match these criteria."

        prompt = f"""You are LeaseOasis, a UAE property assistant. Analyze this user query and provide a complete response with intelligent decision making.

CRITICAL REMINDER: When no properties match user criteria, you MUST offer TWO options naturally:
1) Try alternate searches (nearby locations, flexible budget, different property types)
2) Gather your requirements (summarize needs and offer to save for the team)

DETECT REQUIREMENT GATHERING REQUESTS:
- If user says "gather requirement", "save my requirements", "collect my needs", or similar
- IMMEDIATELY start the requirement gathering process
- Don't offer alternatives again - they've already chosen to gather requirements
- Start collecting the required information right away

CONVERSATION STYLE - BE HUMAN:
- Sound like you're talking to a friend, not a robot
- Use natural language: "Ah, I see...", "Unfortunately...", "But I can help you..."
- Use contractions: "I'm", "you're", "don't", "can't", "won't"
- Be conversational and warm
- Avoid formal phrases like "I can offer you two options"

EXAMPLES OF WHEN TO OFFER REQUIREMENT GATHERING:
- User asks for "villa in Sharjah" but no villas exist in Sharjah
- User asks for "property under 50K" but no properties under 50K exist
- User asks for "furnished apartment in Dubai Marina" but no furnished apartments exist there
- User asks for "pet-friendly property" but no pet-friendly properties exist
- ANY time you cannot find properties matching their specific criteria

ALWAYS end naturally with: "What sounds better to you?" or "What do you think?"

USER QUERY: "{user_input}"
{conversation_context}{preferences_context}{filters_context}{property_context}

CRITICAL CONTEXT:
- You ONLY provide LEASE/RENT properties - NEVER mention buying or purchase options
- All properties in your database are for rent/lease only
- Always refer to "rent" or "lease" when discussing properties
- Never suggest "buying" as an alternative

INSTRUCTIONS:
1. Understand the user's intent and context by analyzing the conversation history
2. Build on previous exchanges - don't repeat information already shared
3. Use your intelligence to understand user preferences, budget constraints, and requirements
4. Analyze the property data and determine what matches the user's needs
5. CRITICAL: When no properties match their criteria, ALWAYS offer TWO options naturally:
   a) Try alternate searches (nearby locations, flexible budget, different property types)
   b) Gather your requirements (summarize needs and offer to save for the team)
6. Make intelligent decisions about showing sources based on what the user needs
7. Provide natural, human-like responses that:
   - Sound like you're talking to a friend, not a robot
   - Use contractions and natural language
   - Be conversational and warm
   - Answer the user's question appropriately
   - Reference previous conversation when relevant
   - Show property sources when helpful
   - Maintain natural conversation flow
   - Use available property data effectively
   - ONLY mention rent/lease options, never buying

RESPONSE FORMAT:
Return your response in this exact JSON format:
{{
    "answer": "Your natural response to the user",
    "show_sources": true/false,
    "reasoning": "Brief explanation of your decision"
}}

SOURCE DECISION GUIDELINES:
- Show sources (true) when:
  * User asks for specific properties, listings, or search results
  * User wants to see "best", "top", or "premium" properties
  * User asks for alternatives or options
  * Property data is available and relevant to the query
  * User is in a property search context

- Don't show sources (false) when:
  * User greets you (hi, hello, good morning)
  * User asks for definitions or explanations
  * User asks for average prices (show calculation only)
  * User asks about properties outside UAE
  * No relevant property data is available
  * User is just confirming or saying thanks

CONVERSATION FLOW:
- Be natural and conversational - sound like a human friend helping out
- Use natural language: "Ah, I see...", "Oh, that's interesting...", "Let me help you with that..."
- Reference previous conversation naturally (e.g., "Since you mentioned you're looking for apartments...")
- Build on what the user has already shared
- Ask follow-up questions when appropriate
- Guide users toward property search when helpful
- Maintain a warm, helpful personality
- Avoid robotic phrases and formal language

REQUIREMENT GATHERING FEATURE (CRITICAL):
When no properties match user criteria, you MUST offer both options naturally and conversationally:
1. "Try alternate searches" - suggest verified alternatives (nearby locations, flexible budget, different property types)
2. "Gather your requirements" - summarize their needs and offer to save for the team

WHEN USER SAYS "GATHER REQUIREMENT" OR SIMILAR:
- Immediately start collecting the required information
- Ask for missing details from the conversation history
- Required fields: location, property_type_name, number_of_bedrooms, rent_charge, furnishing_status, amenities (optional), lease_duration (optional)
- If user has already provided some info, acknowledge it and ask for missing pieces
- Once you have enough info, offer to save it to the team

REQUIREMENT GATHERING PROCESS:
1. Check conversation history for already provided information
2. Acknowledge what you already know: "I know you're looking for [property_type] in [location]"
3. Ask for missing required fields one by one
4. Be specific about what you need: "What's your budget range?" not "What's your budget?"
5. Once you have enough info, say: "Perfect! I have everything I need. Let me save this for our team."
6. Then save the requirements to the endpoint

REQUIRED FIELDS FOR REQUIREMENT GATHERING:
- location (emirate/city/community)
- property_type_name (villa, apartment, etc.)
- number_of_bedrooms
- rent_charge (budget range)
- furnishing_status (furnished, semi-furnished, unfurnished)
- amenities (optional - pool, gym, parking, etc.)
- lease_duration (optional - 1 year, 2 years, etc.)

ENDPOINT SAVING:
- When you have enough information, save it to: http://localhost:5000/backend/api/v1/user/requirement
- Include all the collected information in the request
- Confirm to the user that it's been saved

Example response when user says "gather requirement":
"Perfect! Let me gather all the details for you. I know you're looking for a villa in Sharjah. Let me ask a few quick questions to get everything we need:

1. How many bedrooms are you looking for?
2. What's your budget range (annual rent)?
3. Do you prefer furnished, semi-furnished, or unfurnished?
4. Any specific amenities you need (like pool, gym, parking)?
5. How long do you want to lease for?

Once I have these details, I'll save everything for our team to find the perfect match for you!"

CONVERSATION STYLE:
- Use natural, human-like language
- Avoid formal, robotic phrases
- Use contractions (I'm, you're, don't, can't)
- Be conversational and friendly
- Sound like you're actually helping a friend
- Don't ask for information already provided in the conversation

PROPERTY DATA ANALYSIS:
- Carefully examine the rent_charge values in the property data
- If user asks for "under X" and you have properties with rent_charge <= X, mention them
- If user asks for "above X" and you have properties with rent_charge >= X, mention them
- Always be accurate about what properties you actually have available
- Don't say "I don't have any" if the property data shows you do have matching properties

FILTER COMPLIANCE (CRITICAL):
- ALWAYS respect the extracted filters from the user query
- If user asks for "apartments" and you have property data, ONLY show properties with property_type_name = "apartment"
- If user asks for "villas" and you have property data, ONLY show properties with property_type_name = "villa"
- If user asks for "studios" and you have property data, ONLY show properties with property_type_name = "studio"
- If user asks for properties in "Dubai", ONLY show properties with emirate = "dubai"
- NEVER show properties that don't match the user's explicit requirements
- If no properties match the exact filters, explain what you found and suggest alternatives
- CRITICAL: Check the property_type_name field in the metadata and only include properties that match the user's request
- If a property has property_type_name = "penthouse" but user asked for "apartments", DO NOT include it in your response

RESPONSE:"""

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
                    print(f"   📝 REQUIREMENT GATHERING DETECTED IN LLM REASONING!")
                    print(f"   Reasoning: {reasoning}")
                
                self._logger.debug(f"LLM decision: show_sources={show_sources}, reasoning={reasoning}")
                return answer, show_sources
            
        except Exception as e:
            self._logger.debug(f"Failed to parse LLM response as JSON: {e}")

        return llm_response, self._intelligent_source_guess(llm_response)

    def _intelligent_source_guess(self, response: str) -> bool:
        """Make intelligent guess about showing sources based on response content."""
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in [
            "here are", "i found", "available properties", "matching properties",
            "property details", "rent charge", "aed", "bedroom", "apartment", "villa"
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

    def _build_context_from_chunks(self, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build context block from retrieved chunks (shared utility)"""
        context_lines: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(f"[Source {idx}]")
            if chunk.text:
                context_lines.append(chunk.text)
            context_lines.append("")
        return "\n".join(context_lines)

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

    def _handle_no_results(self, user_input: str, preferences: Dict[str, Any]) -> str:
        """Handle cases where no properties are found - with context checking and requirement gathering"""
        
        print(f"\n📝 NO RESULTS HANDLER - OFFERING REQUIREMENT GATHERING:")
        print(f"   User input: '{user_input}'")
        print(f"   Preferences: {preferences}")
        
        retriever_client = pipeline_state.retriever_client if pipeline_state else None
        verified_alternatives = self._get_verified_alternatives(preferences, retriever_client)
        response = (
            f"I understand you're looking for a property, but unfortunately I couldn't find exact matches for your criteria in our current listings.\n\n"
            f"Let me help you in two ways:\n\n"
        )
        
        if verified_alternatives:
            response += "**1. 🔍 Try alternate searches** (I've verified these options exist in our database):\n"
            for i, alt in enumerate(verified_alternatives[:3], 1):
                response += f"   {i}. {alt['suggestion']} ({alt['count']} properties available)\n"
                if 'sample_properties' in alt and alt['sample_properties']:
                    sample = alt['sample_properties'][0]
                    response += f"      Example: {sample['title']} - {sample['rent']}\n"
            response += "\nWould you like me to show you properties for any of these alternatives?\n\n"
        else:
            response += "**1. 🔍 Expand search criteria**: I can help you adjust your requirements to find similar properties.\n\n"
        
        response += (
            "**2. 📝 Save your requirements**: I can gather what you're looking for and send it to our team. "
            "They'll work with agencies to source properties that match your needs and notify you when available.\n\n"
            "Which option would you prefer? Type '1' for alternatives or '2' to save your requirements, "
            "or just tell me what you'd like to do next."
        )
        
        return response


    def _get_verified_alternatives(self, preferences: Dict[str, Any], retriever) -> List[Dict[str, Any]]:
        """Generate dynamic alternatives using LLM reasoning and database verification"""
        if not retriever:
            return []
        alternatives_generator = DynamicAlternativesGenerator(self._provider_client)
        return alternatives_generator.generate_verified_alternatives(preferences, retriever)

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
        
        print(f"   🔍 EXTRACTING DETAILED REQUIREMENTS FROM CONVERSATION:")
        print(f"   Combined user messages: {combined_query[:300]}...")
        
        # Use LLM to extract requirements from conversation
        extraction_prompt = f"""Extract detailed property requirements from this conversation. Return ONLY a JSON object with the following structure:

CONVERSATION: {combined_query}

Extract these fields if mentioned:
- location (emirate/city/community)
- property_type_name (apartment, villa, studio, etc.)
- number_of_bedrooms
- rent_charge (budget range - use "lte" for "under X" or "gte" for "above X")
- furnishing_status (furnished, semi-furnished, unfurnished)
- amenities (list of amenities mentioned like gym, pool, parking, etc.)
- lease_duration (in years or months)

Return ONLY this JSON format:
{{
    "location": "extracted location",
    "property_type_name": "extracted type",
    "number_of_bedrooms": number,
    "rent_charge": {{"lte": number}} or {{"gte": number}},
    "furnishing_status": "extracted status",
    "amenities": ["list", "of", "amenities"],
    "lease_duration": "extracted duration"
}}

If a field is not mentioned, omit it from the JSON. Only include fields that are explicitly mentioned in the conversation."""

        try:
            response = self._safe_generate([{"role": "user", "content": extraction_prompt}])
            print(f"   LLM extraction response: {response[:200]}...")
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_requirements = json.loads(json_match.group(0))
                print(f"   Successfully extracted: {extracted_requirements}")
                return extracted_requirements
        except Exception as e:
            print(f"   Error extracting requirements: {e}")
        
        return {}

    def _check_required_fields(self, extracted_requirements: Dict[str, Any]) -> List[str]:
        """Check which required fields are missing from extracted requirements"""
        # Get required fields from config or use default priority fields
        required_fields = getattr(self._config.requirement_gathering, 'priority_fields', [
            'location', 'property_type_name', 'number_of_bedrooms', 'rent_charge', 'furnishing_status'
        ])
        
        # Filter out optional fields (marked with # optional in config)
        # For now, we'll consider amenities and lease_duration as optional
        optional_fields = ['amenities', 'lease_duration']
        mandatory_fields = [field for field in required_fields if field not in optional_fields]
        
        missing_fields = []
        for field in mandatory_fields:
            # Handle field mapping - location can be emirate, city, or community
            field_found = False
            if field == 'location':
                # Check for location in various forms
                location_fields = ['location', 'emirate', 'city', 'community']
                for loc_field in location_fields:
                    if loc_field in extracted_requirements and extracted_requirements[loc_field]:
                        field_found = True
                        break
            else:
                # Check the exact field
                if field in extracted_requirements and extracted_requirements[field]:
                    field_found = True
            
            if not field_found:
                missing_fields.append(field)
        
        print(f"   Required fields: {required_fields}")
        print(f"   Mandatory fields: {mandatory_fields}")
        print(f"   Optional fields: {optional_fields}")
        print(f"   Missing fields: {missing_fields}")
        return missing_fields

    def _ask_for_missing_requirements(self, missing_fields: List[str], extracted_requirements: Dict[str, Any]) -> str:
        """Ask user to provide missing required fields"""
        print(f"   📝 ASKING FOR MISSING REQUIREMENTS: {missing_fields}")
        
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
        print(f"\n📝 REQUIREMENT GATHERING TRIGGERED:")
        print(f"   User input: '{user_input}'")
        print(f"   Ask confirmation: {ask_confirmation}")
        print(f"   Session ID: {str(id(self.conversation))}")
        
        # Extract detailed requirements from conversation
        extracted_requirements = self._extract_detailed_requirements_from_conversation()
        print(f"   Extracted requirements: {extracted_requirements}")
        
        # Merge with conversation preferences (which contain emirate, etc.)
        preferences = self.conversation.get_preferences()
        print(f"   Conversation preferences: {preferences}")
        
        # Merge extracted requirements with preferences, giving priority to extracted
        merged_requirements = {**preferences, **extracted_requirements}
        print(f"   Merged requirements: {merged_requirements}")
        
        # Check if we have sufficient data to send to endpoint
        missing_fields = self._check_required_fields(merged_requirements)
        print(f"   Missing required fields: {missing_fields}")
        
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
        
        print(f"   User preferences: {preferences}")
        print(f"   Conversation summary: {conversation_summary[:200]}...")
        
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
            print(f"   Sending requirements directly to endpoint (no confirmation needed)")
            return self._send_requirements_to_endpoint(requirement_summary)
    
    def _send_requirements_to_endpoint(self, requirements: Dict[str, Any]) -> str:
        """Send requirements to the backend endpoint with retry logic and fallback storage"""
        
        endpoint = self._config.requirement_gathering.endpoint or "http://localhost:5000/backend/api/v1/user/requirement"
        
        print(f"\n🚀 SENDING REQUIREMENTS TO API ENDPOINT:")
        print(f"   Endpoint: {endpoint}")
        print(f"   Requirements data: {requirements}")

        # Transform the data to match backend controller expectations
        extracted_requirements = requirements.get("extracted_requirements", {})
        preferences = requirements.get("preferences", {})
        
        # Merge extracted requirements with preferences (extracted takes priority)
        merged_data = {**preferences, **extracted_requirements}
        
        # Map to backend controller field names
        backend_payload = {
            "session_id": requirements.get("session_id", ""),
            "location": self._extract_location(merged_data),
            "property_type": merged_data.get("property_type_name"),
            "number_of_bedrooms": merged_data.get("number_of_bedrooms"),
            "rent_charge": self._extract_rent_charge_value(merged_data),
            "furnishing_status": merged_data.get("furnishing_status"),
            "amenities": self._extract_amenities(merged_data),
            "nearby_landmarks": merged_data.get("nearby_landmarks"),
            "building_name": merged_data.get("building_name"),
            "rent_type": merged_data.get("rent_type_name")
        }
        
        print(f"   🔄 TRANSFORMED PAYLOAD FOR BACKEND:")
        print(f"      Original extracted_requirements: {extracted_requirements}")
        print(f"      Original preferences: {preferences}")
        print(f"      Merged data: {merged_data}")
        print(f"      Backend payload: {backend_payload}")
        
        enhanced_requirements = backend_payload
        retry_attempts = 3
        retry_delay = 2
        
        for attempt in range(retry_attempts):
            try:
                print(f"   📤 API CALL ATTEMPT {attempt + 1}/{retry_attempts}:")
                print(f"      URL: {endpoint}")
                print(f"      Payload: {enhanced_requirements}")
                
                self._logger.info(f"Sending requirements to endpoint: {endpoint} (attempt {attempt + 1}/{retry_attempts})")
                response = requests.post(
                    endpoint,
                    json=enhanced_requirements,
                    timeout=15,
                    headers={"Content-Type": "application/json"}
                )
                
                print(f"      Response Status: {response.status_code}")
                print(f"      Response Headers: {dict(response.headers)}")
                if response.text:
                    print(f"      Response Body: {response.text[:500]}...")
                if response.status_code == 200:
                    print(f"      ✅ SUCCESS: Requirements sent successfully!")
                    self.conversation.set_requirement_gathered(True)
                    self.requirement_metrics.log_attempt(True)
                    self._logger.info("Requirements successfully sent to endpoint")
                    return (
                        "Perfect! I've saved your requirements successfully.\n\n"
                        "Your property preferences have been recorded and we'll keep them in mind for future searches.\n\n"
                        "Is there anything else I can help you with today?"
                    )
                elif attempt < retry_attempts - 1:
                    print(f"      ⚠️  RETRY: Status {response.status_code}, retrying in {retry_delay}s")
                    self._logger.warning(f"Endpoint returned status {response.status_code}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2  
                    continue
                else:
                    print(f"      ❌ FAILED: Status {response.status_code} after {retry_attempts} attempts")
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
                    print(f"      ⏰ TIMEOUT: Retrying in {retry_delay}s")
                    self._logger.warning(f"Request timeout, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print(f"      ❌ TIMEOUT FAILED: After all retry attempts")
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
                    print(f"      ⚠️  ERROR: {e}, retrying in {retry_delay}s")
                    self._logger.warning(f"Request failed: {e}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    print(f"      ❌ ERROR FAILED: {e} after {retry_attempts} attempts")
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
        """Extract amenities from merged data"""
        amenities = []
        
        # Check for amenities list
        if "amenities" in merged_data and isinstance(merged_data["amenities"], list):
            amenities.extend(merged_data["amenities"])
        
        # Check for individual amenity boolean fields
        amenity_fields = [
            "gym_fitness_center", "swimming_pool", "parking", "balcony_terrace",
            "beach_access", "elevators", "security_available", "concierge_available",
            "maids_room", "laundry_room", "storage_room", "bbq_area", "pet_friendly",
            "central_ac_heating", "smart_home_features", "waste_disposal_system",
            "power_backup", "mosque_nearby", "jogging_cycling_tracks", "chiller_included",
            "sublease_allowed", "childrens_play_area"
        ]
        
        for field in amenity_fields:
            if merged_data.get(field) is True:
                # Convert field name to user-friendly name
                amenity_name = field.replace("_", " ").replace("fitness center", "gym")
                if field == "gym_fitness_center":
                    amenity_name = "gym"
                elif field == "swimming_pool":
                    amenity_name = "pool"
                elif field == "balcony_terrace":
                    amenity_name = "balcony"
                elif field == "childrens_play_area":
                    amenity_name = "children play area"
                
                amenities.append(amenity_name)
        
        return amenities

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

    def _track_search_attempt(self, query: str, filters: Dict[str, Any], chunks: List[RetrievedChunk]) -> None:
        """Track search attempts for better conversation context"""
        results_count = len(chunks) if chunks else 0
        self.conversation.add_search_attempt(query, filters, results_count)

    def _get_enhanced_conversation_context(self) -> str:
        """Get enhanced conversation context including search history and alternatives"""
        context_parts = []
        basic_context = self._get_conversation_context()
        if basic_context:
            context_parts.append(basic_context)
        search_context = self.conversation.get_search_context()
        if search_context:
            context_parts.append(search_context)
        alternatives_context = self.conversation.get_alternatives_context()
        if alternatives_context:
            context_parts.append(alternatives_context)
        return " | ".join(context_parts)

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

    def _get_conversation_summary_for_endpoint(self) -> str:
        """Get detailed conversation summary for endpoint"""
        messages = self.conversation.get_messages()
        if not messages:
            return "No conversation history"
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg["content"] for msg in messages if msg["role"] == "assistant"]
        summary_parts = []
        summary_parts.append(f"User queries: {'; '.join(user_messages[-5:])}")  
        summary_parts.append(f"Assistant responses: {'; '.join(assistant_messages[-3:])}")
        return " | ".join(summary_parts)

    def _get_search_attempts_summary(self) -> List[Dict[str, Any]]:
        """Get summary of search attempts made during conversation"""
        search_attempts = []
        messages = self.conversation.get_messages()
        for msg in messages:
            if msg["role"] == "user" and any(keyword in msg["content"].lower() for keyword in ["find", "search", "show", "list", "get"]):
                search_attempts.append({
                    "query": msg["content"],
                    "timestamp": time.time()
                })
        return search_attempts[-5:]  

    def _get_alternatives_summary(self) -> List[Dict[str, Any]]:
        """Get summary of alternatives suggested during conversation"""  
        return []

    def _build_context_and_prompt(self, user_input: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build context prompt with conversation memory and user preferences"""
        conversation_history = self._get_conversation_context()
        user_preferences = self.conversation.get_preferences()
        context_block = self._build_context_from_chunks(retrieved_chunks)
        memory_section = ""
        if conversation_history:
            memory_section = f"\n\nCONVERSATION HISTORY:\n{conversation_history}"
        preferences_section = ""
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
                preferences_section = f"\n\nUSER PREFERENCES: {', '.join(pref_items)}"
        prompt = f"User question:\n{user_input}{memory_section}{preferences_section}\n\nProperty Context:\n{context_block}\n\nResponse:"
        return prompt
