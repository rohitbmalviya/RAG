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

# Import pipeline_state for alternative suggestions
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
            logger.info(
                f"Requirement gathering stats: {self.successful_sends}/{self.total_attempts} successful"
            )

class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.user_preferences: Dict[str, Any] = {}
        self.conversation_summary: str = ""
        self.requirement_gathered: bool = False

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences from conversation"""
        self.user_preferences.update(preferences)

    def get_preferences(self) -> Dict[str, Any]:
        return self.user_preferences

    def set_requirement_gathered(self, gathered: bool):
        self.requirement_gathered = gathered

    def is_requirement_gathered(self) -> bool:
        return self.requirement_gathered

    def get_conversation_summary(self) -> str:
        return self.conversation_summary

    def set_conversation_summary(self, summary: str):
        self.conversation_summary = summary

LLM_PROVIDERS: Dict[str, Type["BaseLLMProvider"]] = {}

# Common identity and introduction text
AGENT_IDENTITY = "LeaseOasis, your friendly UAE property assistant"
AGENT_DESCRIPTION = "UAE property leasing assistant"
AGENT_SPECIALIZATION = "help people find the perfect place to lease in Dubai, Abu Dhabi, and other UAE cities"
# Common instruction patterns
JSON_ONLY_INSTRUCTION = "Return ONLY a JSON"
NO_ADDITIONAL_TEXT = "no additional text"
BRIEF_EXPLANATION = "Brief explanation"
# Common response patterns
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

# Common JSON extraction patterns
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

# Common prompt templates
def _build_query_classification_template() -> str:
    """Build the query classification template with proper formatting"""
    # Use string replacement to avoid f-string brace conflicts
    template = """Classify this user query for a {agent_description}.

QUERY: "{query}"

Classify into ONE of these categories and return ONLY a JSON object:

{{
    "category": "greeting|general_knowledge|best_property|average_price|property_search|outside_uae|general|conversation_response",
    "confidence": 0.95,
    "reasoning": "{brief_explanation} of classification"
}}

CATEGORIES:
- "greeting": Simple greetings (hi, hello, good morning, etc.)
- "general_knowledge": Questions about property terms, definitions, explanations
- "best_property": Queries asking for "best", "top", "premium", "featured" properties
- "average_price": Queries asking for average/typical/mean prices or costs
- "property_search": Queries looking for specific properties to lease/rent
- "outside_uae": Queries about properties outside UAE (non-UAE locations)
- "general": General property-related queries that don't fit other categories
- "conversation_response": Responses to conversation flow (1, 2, yes, no, etc.)

RULES:
- Use high confidence (0.8+) for clear matches
- Use lower confidence (0.6-0.8) for ambiguous cases
- Consider context and intent, not just keywords
- For conversation responses, check for simple responses like "1", "2", "yes", "no"

Output JSON:"""
    
    # Replace only the constants, leave {query} for later formatting
    return template.replace("{agent_description}", AGENT_DESCRIPTION).replace("{brief_explanation}", BRIEF_EXPLANATION)
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
    "type": "location|budget|property_type|amenities|furnishing",
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

RULES:
- Only suggest alternatives that make sense for UAE properties
- For location: suggest nearby communities or adjacent emirates
- For budget: suggest ±20% flexibility
- For property type: suggest similar but different types
- Be practical and realistic
- Focus on what would actually help find more properties

Output JSON array:"""
COMPLETE_SYSTEM_PROMPT = f"""You are {AGENT_IDENTITY} with conversational memory.

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

CONVERSATION FLOW:
- Start with warm greetings and open-ended questions
- Guide users through property search step by step
- When users are vague, ask ONE clarifying question at a time
- Build excitement about properties that match their needs
- End conversations warmly with clear next steps

QUERY HANDLING:

GREETINGS (Sources: Empty):
- Respond warmly to greetings (hi, hello, good morning)
- Ask an engaging follow-up question about their property search
- Examples: 'Which UAE city are you interested in?', 'What brings you here today?', 'Are you looking for your first property or moving to a new area?'
- NO property details should be shown

GENERAL KNOWLEDGE QUERIES (Sources: Empty):
- Answer questions about property terms (What is an apartment? What is holiday home ready?)
- Use web search if needed for current information
- Provide clear, helpful explanations
- Guide back to property search when appropriate
- NO property listings should be shown

BEST PROPERTY QUERIES (Sources: Show):
- Prioritize properties in this EXACT order:
  1. premiumBoostingStatus: 'Active' AND carouselBoostingStatus: 'Active' AND bnb_verification_status: 'verified'
  2. bnb_verification_status: 'verified'
  3. carouselBoostingStatus: 'Active'
  4. premiumBoostingStatus: 'Active'
- Explain WHY these are the 'best' properties
- Show property details with sources
- Make users excited about these premium options

AVERAGE PRICE QUERIES (Sources: Empty):
- Calculate average rent_charge from retrieved context
- Provide the average value clearly
- Give context about the calculation (e.g., 'based on 25 properties')
- NO individual property details should be shown
- Keep sources empty

PROPERTY SEARCH QUERIES (Sources: Show):
- Use conversation memory to understand what user wants
- If user gives vague location (e.g., 'Dubai'), ask for specific area
- If user doesn't specify budget, ask about their range
- If user doesn't specify bedrooms, ask about their needs
- Only show properties when you have enough context
- Present properties in an exciting, personalized way
- Always show sources for property details

NO MATCHES FOUND:
- Acknowledge their specific requirements with empathy
- Offer TWO clear options:
  1. 'Try alternate searches' - suggest verified alternatives (nearby locations, flexible budget, different property types)
  2. 'Gather your requirements' - summarize their needs and offer to save for the team
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

UAE ONLY POLICY (Sources: Empty):
- Politely redirect non-UAE property queries
- Explain you specialize in UAE properties
- Suggest UAE alternatives
- Keep sources empty

IDENTITY QUERIES (Sources: Empty):
- Respond: 'I am {AGENT_IDENTITY}. I {AGENT_SPECIALIZATION}.'
- Ask how you can help with their property search
- Keep sources empty

SOURCES RULES:
- ALWAYS show sources for: property search results, best property queries, specific property details
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
        
        # Fallback to simple keyword matching if LLM fails
        return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Fallback classification using simple patterns"""
        query_lower = query.lower().strip()
        
        # Simple fallback patterns
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


# DynamicPreferenceExtractor removed - using query_processor.py instead

class DynamicAlternativesGenerator:
    """Dynamic LLM-based alternatives generation"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def generate_verified_alternatives(self, preferences: Dict[str, Any], retriever) -> List[Dict[str, Any]]:
        """Generate alternatives using LLM reasoning and verify with database"""
        if not preferences:
            return []
        
        # Use LLM to generate alternative suggestions
        alternatives = self._generate_alternatives_with_llm(preferences)
        
        # Verify alternatives exist in database
        verified_alternatives = []
        for alt in alternatives:
            if self._verify_alternative_exists(alt, retriever):
                verified_alternatives.append(alt)
        
        return verified_alternatives[:3]  # Return top 3 verified alternatives
    
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
            
            # Test query with generic terms to avoid too specific searches
            test_queries = [
                suggestion,
                f"property in {suggestion}",
                "available properties"
            ]
            
            for query in test_queries:
                try:
                    test_chunks = retriever.retrieve(query, filters=filters, top_k=15)
                    
                    if test_chunks and len(test_chunks) >= 3:
                        # Verify chunks have actual property data
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
        return FALLBACK_GREETINGS[0]  # Return first greeting as fallback
    
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
        
        # Initialize dynamic components
        self.query_classifier = DynamicQueryClassifier(self._provider_client)
        
        # Initialize metrics for requirement gathering
        self.requirement_metrics = RequirementGatheringMetrics()

        # Set consolidated system instructions
        self.conversation.add_message("system", COMPLETE_SYSTEM_PROMPT)

    def chat(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Process user query using dynamic classification and handling."""
        user_input = user_input.strip()

        # Use dynamic query classification
        classification = self.query_classifier.classify_query(user_input)
        category = classification.get("category", "general")
        confidence = classification.get("confidence", 0.5)
        
        self._logger.debug(f"Query classified as: {category} (confidence: {confidence})")

        # Handle based on dynamic classification
        if category == "conversation_response":
            return self._handle_conversation_response(user_input, retrieved_chunks)
        elif category == "greeting":
            return self._handle_greeting(user_input)
        elif category == "outside_uae":
            return self._handle_outside_uae_query(user_input)
        elif category == "general_knowledge":
            return self._handle_general_knowledge_query(user_input)
        elif category == "best_property":
            return self._handle_best_property_query(user_input, retrieved_chunks)
        elif category == "average_price":
            return self._handle_average_price_query(user_input, retrieved_chunks)
        elif category == "property_search":
            return self._handle_property_query(user_input, retrieved_chunks)
        else:
            return self._handle_general_query(user_input)


    def _handle_conversation_response(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None):
        """Handle user responses to conversation flow (1, 2, yes, no)"""
        user_input = user_input.lower().strip()
        
        # Get recent conversation context with proper message parsing
        recent_messages = self.conversation.get_messages()[-4:]  # Last 4 messages
        last_assistant_message = ""
        original_query_context = ""
        
        for msg in reversed(recent_messages):
            if msg["role"] == "assistant":
                last_assistant_message = msg["content"]
                break
        
        # Get original user query for context
        for msg in reversed(recent_messages):
            if msg["role"] == "user" and msg["content"] not in ["1", "2", "yes", "no"]:
                original_query_context = msg["content"]
                break
        
        # Handle requirement gathering confirmation
        if user_input in ["yes", "save"] and "save these requirements" in last_assistant_message:
            return self._send_requirements_to_endpoint({
                "user_query": " ".join([msg["content"] for msg in recent_messages if msg["role"] == "user"]),
                "preferences": self.conversation.get_preferences(),
                "conversation_summary": self._summarize_conversation_context(),
                "session_id": str(id(self.conversation)),
                "timestamp": time.time()
            })
        
        elif user_input == "no" and "save these requirements" in last_assistant_message:
            return (
                "No problem! Let's continue exploring other options. "
                "Would you like to try some alternative searches, or would you prefer to adjust your criteria? "
                "I'm here to help you find the perfect property!"
            )
        
        # Handle alternative selection with proper boosting
        elif user_input in ["1", "option 1", "first", "alternatives"] and "Try alternate searches" in last_assistant_message:
            preferences = self.conversation.get_preferences()
            # Apply proper boosting filters based on original query context
            enhanced_filters = preferences.copy()
            # If original query was about "best" properties, apply boosting
            if any(keyword in original_query_context.lower() for keyword in ["best", "top", "premium", "recommended"]):
                enhanced_filters.update({
                    "bnb_verification_status": "verified",
                    "premiumBoostingStatus": "Active"
                })
            # Get verified alternatives with enhanced filters
            verified_alternatives = self._get_verified_alternatives(enhanced_filters, pipeline_state.retriever_client)
            
            if verified_alternatives:
                # Show properties for the first alternative
                alt = verified_alternatives[0]
                try:
                    alt_filters = alt['filters'].copy()
                    alt_filters.update(enhanced_filters)  # Merge with enhanced filters
                    
                    alt_chunks = pipeline_state.retriever_client.retrieve(
                        f"properties {alt['suggestion']}", 
                        filters=alt_filters, 
                        top_k=5
                    ) 
                    if alt_chunks:
                        context_prompt = self._build_context_and_prompt(f"Show me {alt['suggestion']}", alt_chunks)
                        response = self._safe_generate([{"role": "user", "content": context_prompt}])
                        self._add_conversation_messages(user_input, response)
                        return response   
                except Exception as e:
                    self._logger.warning(f"Failed to retrieve alternative properties: {e}")
                    return (
                        "I encountered an issue while searching for alternative properties. "
                        "Let me try a different approach or you can adjust your search criteria."
                    )
            
            return (
                "Let me search for alternative properties based on your preferences. "
                "I'll look for nearby locations and similar options that might work for you."
            )
        
        # Handle requirement gathering selection
        elif user_input in ["2", "option 2", "second", "save"] and ("Gather your requirements" in last_assistant_message or "Save your requirements" in last_assistant_message):
            return self.gather_requirements_and_send("User requested requirement gathering", ask_confirmation=True)
        
        # Default response for unclear conversation responses
        return (
            "I want to make sure I understand what you'd prefer. Could you tell me more specifically what you'd like to do? "
            "For example, you could say 'show me alternatives' or 'save my requirements' or ask me a new question about properties."
        )

    def _handle_greeting(self, user_input: str) -> str:
        """Handle greeting messages using dynamic response generation"""
        # Use dynamic response generator for greetings
        response_generator = DynamicResponseGenerator(self._provider_client)
        response = response_generator.generate_greeting_response(user_input, self.conversation.get_messages())
        self._add_conversation_messages(user_input, response)
        return response

    def _handle_outside_uae_query(self, user_input: str) -> str:
        """Handle queries about properties outside UAE"""
        response = (
            f"I'm sorry, but I can only assist with properties in the UAE. I specialize in helping you find great leasing options in Dubai, Abu Dhabi, Sharjah, and other UAE cities. "
            "Would you like to explore properties in any of these UAE locations instead?"
        )
        self._add_conversation_messages(user_input, response)
        return response

    def _handle_general_knowledge_query(self, user_input: str) -> str:
        """Handle general property knowledge queries using dynamic response generation"""
        try:
            # Use dynamic response generator for general knowledge
            response_generator = DynamicResponseGenerator(self._provider_client)
            response = response_generator.generate_general_knowledge_response(user_input, self.conversation.get_messages())
                
        except Exception as e:
            self._logger.warning(f"Error handling general knowledge query: {e}")
            response = "I can help you with UAE property-related questions. Would you like to search for properties instead?"
        
        self._add_conversation_messages(user_input, response)
        return response

    def _handle_best_property_query(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Handle 'best property' queries with proper prioritization"""
        if retrieved_chunks:
            context_prompt = self._build_context_and_prompt(user_input, retrieved_chunks)
            response = self._safe_generate([{"role": "user", "content": context_prompt}])
        else:
            # No results found - offer alternatives
            preferences = self.conversation.get_preferences()
            response = self._handle_no_results(user_input, preferences)

        self._add_conversation_messages(user_input, response)
        return response

    def _handle_average_price_query(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Handle average price queries with proper calculation and no property sources"""
        
        if not retrieved_chunks:
            return (
                "I don't have enough data to calculate the average price for your query. "
                "Could you please specify the location and property type you're interested in?"
            )
        
        # Calculate average from retrieved chunks
        avg_price = self._calculate_average_price(retrieved_chunks, "rent_charge")
        
        if avg_price:
            # Extract location context from chunks for better response
            locations = set()
            property_types = set()
            
            for chunk in retrieved_chunks:
                if chunk.metadata.get("community"):
                    locations.add(chunk.metadata.get("community"))
                if chunk.metadata.get("emirate"):
                    locations.add(chunk.metadata.get("emirate"))
                if chunk.metadata.get("property_type_name"):
                    property_types.add(chunk.metadata.get("property_type_name"))
            
            location_str = ", ".join(list(locations)[:2]) if locations else "the specified area"
            type_str = ", ".join(list(property_types)[:2]) if property_types else "properties"
            
            response = (
                f"The average rent for {type_str} in {location_str} is "
                f"AED {avg_price:,.0f} per year, based on {len(retrieved_chunks)} available properties."
            )
            
            # Add additional insights if possible
            prices = [float(chunk.metadata.get("rent_charge", 0)) for chunk in retrieved_chunks 
                     if chunk.metadata.get("rent_charge")]
            if prices:
                min_price = min(prices)
                max_price = max(prices)
                response += f"\n\nPrice range: AED {min_price:,.0f} - {max_price:,.0f} per year."
        else:
            response = "I couldn't calculate the average price from the available data for your specified criteria."
        
        # Important: Don't show property sources for average price queries
        self._add_conversation_messages(user_input, response)
        return response

    def _calculate_average_price(self, chunks: List[RetrievedChunk], field: str = "rent_charge") -> Optional[float]:
        """Calculate average price from retrieved chunks with better error handling"""
        prices = []
        
        for chunk in chunks:
            price = chunk.metadata.get(field)
            if price is not None:
                try:
                    # Handle different price formats
                    if isinstance(price, str):
                        # Remove commas and convert
                        price_val = float(price.replace(",", ""))
                    else:
                        price_val = float(price)
                    
                    # Sanity check - prices should be reasonable for UAE
                    if 10000 <= price_val <= 10000000:  # AED 10k to 10M per year
                        prices.append(price_val)
                except (ValueError, TypeError):
                    continue
        
        return sum(prices) / len(prices) if prices else None

    def _handle_property_query(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Handle property search queries using query_processor for preference extraction"""
        # Import locally to avoid circular import
        from .query_processor import extract_filters_with_llm_context_aware
        # Extract user preferences using the comprehensive query_processor
        preferences = extract_filters_with_llm_context_aware(user_input, self)
        self.conversation.update_preferences(preferences)

        if retrieved_chunks:
            context_prompt = self._build_context_and_prompt(user_input, retrieved_chunks)
            response = self._safe_generate([{"role": "user", "content": context_prompt}])
        else:
            # No results found - offer alternatives or gather requirements
            response = self._handle_no_results(user_input, preferences)

        self._add_conversation_messages(user_input, response)
        return response

    def _handle_general_query(self, user_input: str) -> str:
        """Handle general queries"""
        response = (
            f"I'm here to help you find properties to lease in the UAE. I can assist with searching for apartments, villas, and other properties in Dubai, Abu Dhabi, and other UAE cities. "
            "What type of property are you looking for, or would you like me to help you with something specific?"
        )
        self._add_conversation_messages(user_input, response)
        return response

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
            
        # Get last 4 messages (2 user + 2 assistant) for context
        recent_messages = messages[-4:]
        context_parts = []
        
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
            context_parts.append(f"{role}: {content}")
            
        return " | ".join(context_parts)



    def _handle_no_results(self, user_input: str, preferences: Dict[str, Any]) -> str:
        """Handle cases where no properties are found - with context checking and requirement gathering"""
        
        # Check verified alternatives by querying vector database
        retriever_client = pipeline_state.retriever_client if pipeline_state else None
        verified_alternatives = self._get_verified_alternatives(preferences, retriever_client)
        
        conversation_summary = self._summarize_conversation_context()
        
        response = (
            f"I understand you're looking for a property, but unfortunately I couldn't find exact matches for your criteria in our current listings.\n\n"
            f"Let me help you in two ways:\n\n"
        )
        
        if verified_alternatives:
            response += "**1. 🔍 Try alternate searches** (I've verified these options exist in our database):\n"
            for i, alt in enumerate(verified_alternatives[:3], 1):
                response += f"   {i}. {alt['suggestion']} ({alt['count']} properties available)\n"
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
        
        # Use dynamic alternatives generator
        alternatives_generator = DynamicAlternativesGenerator(self._provider_client)
        return alternatives_generator.generate_verified_alternatives(preferences, retriever)

    def _summarize_conversation_context(self) -> str:
        """Summarize conversation for requirement gathering"""
        messages = self.conversation.get_messages()
        preferences = self.conversation.get_preferences()
        
        if not messages:
            return "No conversation history available."
        
        # Extract key conversation points
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        
        summary_parts = []
        summary_parts.append(f"User discussed: {'; '.join(user_messages[-3:])}")  # Last 3 user messages
        
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

    def gather_requirements_and_send(self, user_input: str, ask_confirmation: bool = True) -> str:
        """Gather requirements and optionally send to endpoint"""
        preferences = self.conversation.get_preferences()
        conversation_summary = self._summarize_conversation_context()
        
        # Build comprehensive requirement summary
        requirement_summary = {
            "user_query": user_input,
            "preferences": preferences,
            "conversation_summary": conversation_summary,
            "session_id": str(id(self.conversation)),
            "timestamp": time.time()
        }
        
        if ask_confirmation:
            response = (
                "I'll summarize what you're looking for:\n\n"
                "**Your Requirements:**\n"
            )
            
            if preferences:
                for key, value in preferences.items():
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
            # Send to endpoint without confirmation
            return self._send_requirements_to_endpoint(requirement_summary)
    
    def _send_requirements_to_endpoint(self, requirements: Dict[str, Any]) -> str:
        """Send requirements to the backend endpoint with retry logic and fallback storage"""
        
        endpoint = self._config.requirement_gathering.endpoint or "http://localhost:5000/backend/api/v1/user/requirement"
        
        # Enhanced requirements payload with conversation context
        enhanced_requirements = {
            "user_query": requirements.get("user_query", ""),
            "preferences": requirements.get("preferences", {}),
            "conversation_summary": requirements.get("conversation_summary", ""),
            "session_id": requirements.get("session_id", ""),
            "timestamp": requirements.get("timestamp", time.time()),
            "conversation_history": self._get_conversation_summary_for_endpoint(),
            "search_attempts": self._get_search_attempts_summary(),
            "alternatives_suggested": self._get_alternatives_summary()
        }
        
        # Add retry logic with exponential backoff
        retry_attempts = 3
        retry_delay = 2
        
        for attempt in range(retry_attempts):
            try:
                self._logger.info(f"Sending requirements to endpoint: {endpoint} (attempt {attempt + 1}/{retry_attempts})")
                response = requests.post(
                    endpoint,
                    json=enhanced_requirements,
                    timeout=15,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    self.conversation.set_requirement_gathered(True)
                    self.requirement_metrics.log_attempt(True)
                    self._logger.info("Requirements successfully sent to endpoint")
                    return (
                        "✅ Perfect! I've saved your requirements and sent them to our team.\n\n"
                        "Our property specialists will:\n"
                        "• Work with local agencies to find matching properties\n"
                        "• Add new listings that meet your criteria\n"
                        "• Notify you when suitable properties become available\n\n"
                        "This helps us understand what users are looking for and improve our platform. "
                        "Is there anything else I can help you with today?"
                    )
                elif attempt < retry_attempts - 1:
                    self._logger.warning(f"Endpoint returned status {response.status_code}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    self._logger.warning(f"Endpoint returned status {response.status_code} after {retry_attempts} attempts")
                    # Save to fallback storage on final failure
                    self._save_to_fallback_storage(enhanced_requirements)
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've noted your requirements, but there was a technical issue saving them to our system. "
                        "Don't worry - I can still help you search for properties or try again later. "
                        "Would you like to continue exploring available options?"
                    )
                    
            except requests.exceptions.Timeout:
                if attempt < retry_attempts - 1:
                    self._logger.warning(f"Request timeout, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
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
                    self._logger.warning(f"Request failed: {e}, retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    self._logger.error(f"Failed to send requirements to endpoint after {retry_attempts} attempts: {e}")
                    self._save_to_fallback_storage(enhanced_requirements)
                    self.requirement_metrics.log_attempt(False)
                    return (
                        "I've carefully noted your requirements! While there was a technical issue with our system, "
                        "I can still help you search for properties with different criteria. "
                        "Would you like to try some alternative searches?"
                    )
        
        # This should never be reached, but just in case
        return (
            "I've carefully noted your requirements! While there was a technical issue with our system, "
            "I can still help you search for properties with different criteria. "
            "Would you like to try some alternative searches?"
        )

    def _save_to_fallback_storage(self, requirements: Dict[str, Any]) -> None:
        """Save requirements to fallback storage when endpoint fails"""
        try:
            # For now, just log the requirements - in production, you might save to a file or database
            fallback_data = {
                "timestamp": time.time(),
                "session_id": requirements.get("session_id", ""),
                "user_query": requirements.get("user_query", ""),
                "preferences": requirements.get("preferences", {}),
                "conversation_summary": requirements.get("conversation_summary", ""),
                "fallback_reason": "endpoint_failure"
            }
            
            self._logger.info(f"Saved requirements to fallback storage: {fallback_data}")
            
            # In a production environment, you might want to:
            # 1. Save to a local file
            # 2. Save to a database table
            # 3. Send to a queue for later processing
            # 4. Send to an alternative endpoint
            
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

    def _get_conversation_summary_for_endpoint(self) -> str:
        """Get detailed conversation summary for endpoint"""
        messages = self.conversation.get_messages()
        if not messages:
            return "No conversation history"
        
        # Get user messages for context
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg["content"] for msg in messages if msg["role"] == "assistant"]
        
        summary_parts = []
        summary_parts.append(f"User queries: {'; '.join(user_messages[-5:])}")  # Last 5 user messages
        summary_parts.append(f"Assistant responses: {'; '.join(assistant_messages[-3:])}")  # Last 3 assistant responses
        
        return " | ".join(summary_parts)

    def _get_search_attempts_summary(self) -> List[Dict[str, Any]]:
        """Get summary of search attempts made during conversation"""
        # Track search attempts in conversation metadata
        search_attempts = []
        messages = self.conversation.get_messages()
        
        for msg in messages:
            if msg["role"] == "user" and any(keyword in msg["content"].lower() for keyword in ["find", "search", "show", "list", "get"]):
                search_attempts.append({
                    "query": msg["content"],
                    "timestamp": time.time()
                })
        
        return search_attempts[-5:]  # Last 5 search attempts

    def _get_alternatives_summary(self) -> List[Dict[str, Any]]:
        """Get summary of alternatives suggested during conversation"""
        # This would track alternatives suggested - for now return empty list
        # In a full implementation, you'd track this in conversation metadata
        return []

    def _build_context_and_prompt(self, user_input: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build context prompt with conversation memory and user preferences"""
        # Get conversation context for short-term memory
        conversation_history = self._get_conversation_context()
        user_preferences = self.conversation.get_preferences()
        
        # Build context block from retrieved chunks
        context_block = self._build_context_from_chunks(retrieved_chunks)
        
        # Build conversation context section
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

        # Build complete prompt with context
        prompt = f"User question:\n{user_input}{memory_section}{preferences_section}\n\nProperty Context:\n{context_block}\n\nResponse:"
        return prompt
