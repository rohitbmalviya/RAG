"""
Minimal, generic LLM client for the RAG Pipeline.

Design Philosophy:
- LLM intelligence over hardcoded logic
- 3 core functions only (no more 50+ specific handlers)
- Generic and reusable for ANY domain
- Context-aware conversation management
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List, Optional, Type
from .config import LLMConfig
from .core.base import BaseLLM
from .models import RetrievedChunk
from .utils import get_logger
from .instructions import SYSTEM_PROMPT, CONTEXT_TEMPLATE, REQUIREMENT_COLLECTION_PROMPT

logger = get_logger(__name__)

# ==================================================================================
# LLM Provider Registry
# ==================================================================================

LLM_PROVIDERS: Dict[str, Type["BaseLLMProvider"]] = {}

def register_llm_provider(name: str):
    """Decorator to register LLM providers"""
    def decorator(cls: Type["BaseLLMProvider"]) -> Type["BaseLLMProvider"]:
        LLM_PROVIDERS[name.lower()] = cls
        return cls
    return decorator

# ==================================================================================
# Base LLM Provider
# ==================================================================================

class BaseLLMProvider:
    """Base class for LLM providers (Google, OpenAI, etc.)"""
    
    def __init__(self, config: LLMConfig, api_key: Optional[str]) -> None:
        self._config = config
        self._api_key = api_key or os.getenv("LLM_MODEL_API_KEY")
        self._logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._setup_client()

    def _setup_client(self) -> None:
        """Setup provider-specific client"""
        pass

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from messages"""
        raise NotImplementedError

# ==================================================================================
# Google LLM Provider
# ==================================================================================

@register_llm_provider("google")
class GoogleLLM(BaseLLMProvider):
    """Google Gemini LLM provider"""
    
    def _setup_client(self) -> None:
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError("google-generativeai package not available") from exc
        
        if self._api_key:
            genai.configure(api_key=self._api_key)
        else:
            self._logger.debug("Google API key missing")
        
        self._model = genai.GenerativeModel(model_name=self._config.model)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Google Gemini"""
        # Convert messages to prompt format
        prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self._config.temperature or 0.3,
                    "max_output_tokens": self._config.max_output_tokens or 2048,
                },
            )
            return response.text
        except Exception as exc:
            self._logger.error(f"Google LLM generation failed: {exc}")
            raise

# ==================================================================================
# OpenAI LLM Provider
# ==================================================================================

@register_llm_provider("openai")
@register_llm_provider("azure_openai")
class OpenAILLM(BaseLLMProvider):
    """OpenAI/Azure OpenAI LLM provider"""
    
    def _setup_client(self) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package not available") from exc

        if not self._api_key:
            self._logger.debug("OpenAI API key missing")

        self._client = OpenAI(api_key=self._api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI"""
        try:
            resp = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature or 0.3,
                max_tokens=self._config.max_output_tokens or 2048,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            self._logger.error(f"OpenAI LLM generation failed: {exc}")
            raise

# ==================================================================================
# Conversation Memory Manager
# ==================================================================================

class Conversation:
    """
    Manages conversation state and memory.
    
    This is GENERIC - works for any domain (properties, cars, jobs, products).
    The LLM handles all domain-specific logic.
    """
    
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.shown_entities: set = set()  # Track shown property/product/job IDs
        self.user_preferences: Dict[str, Any] = {}  # Extracted filters/preferences
        self.interaction_count: int = 0
        self.last_activity_time: float = time.time()
        self.last_search_results: List[str] = []  # Track property IDs from last search for progressive filtering
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.last_activity_time = time.time()
        
        if role == "user":
            self.interaction_count += 1
        
        # Keep only last 20 messages to prevent context overflow
        if len(self.messages) > 20:
            # Keep system prompt (first message) + last 19 messages
            self.messages = [self.messages[0]] + self.messages[-19:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.messages
    
    def track_shown_entity(self, entity_id: str) -> None:
        """Track that an entity (property/product/job) was shown to user"""
        self.shown_entities.add(entity_id)
    
    def was_shown(self, entity_id: str) -> bool:
        """Check if entity was already shown"""
        return entity_id in self.shown_entities
    
    def get_shown_entities(self) -> List[str]:
        """Get list of shown entity IDs"""
        return list(self.shown_entities)
    
    def update_preferences(self, preferences: Dict[str, Any]) -> None:
        """Update user preferences from conversation"""
        self.user_preferences.update(preferences)
    
    def store_search_results(self, property_ids: List[str]) -> None:
        """Store property IDs from current search for progressive filtering"""
        self.last_search_results = property_ids
    
    def get_last_search_results(self) -> List[str]:
        """Get property IDs from last search"""
        return self.last_search_results
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        return self.user_preferences
    
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation for context"""
        if len(self.messages) <= 1:  # Only system prompt
            return "New conversation - no previous context"
        
        # Get all non-system messages for better context
        conversation_messages = [msg for msg in self.messages if msg["role"] != "system"]
        
        if not conversation_messages:
            return "New conversation - no previous context"
        
        # Build conversation history
        summary_parts = []
        
        # Add conversation count for context
        user_messages = [msg for msg in conversation_messages if msg["role"] == "user"]
        summary_parts.append(f"Conversation has {len(user_messages)} user queries so far.")
        
        # Show recent conversation (last 4 exchanges = 8 messages)
        recent = conversation_messages[-8:] if len(conversation_messages) > 8 else conversation_messages
        
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            
            # For user messages, show full content (they're usually short)
            # For assistant messages, truncate if very long
            if msg["role"] == "assistant" and len(content) > 200:
                content = content[:200] + "..."
            
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def is_expired(self, ttl_seconds: int = 1800) -> bool:
        """Check if conversation has expired (30 minutes default)"""
        return (time.time() - self.last_activity_time) > ttl_seconds
    
    def clear(self) -> None:
        """Clear conversation (keep system prompt)"""
        system_prompt = self.messages[0] if self.messages else None
        self.messages = [system_prompt] if system_prompt else []
        self.shown_entities.clear()
        self.user_preferences.clear()
        self.interaction_count = 0

# ==================================================================================
# Main LLM Client
# ==================================================================================

class LLMClient(BaseLLM):
    """
    Main LLM client - handles all LLM interactions.
    
    Design: Minimal and generic. The LLM does all the work.
    This class just:
    1. Manages conversation state
    2. Builds prompts with context
    3. Calls LLM provider
    
    NO hardcoded logic for specific scenarios!
    """
    
    def __init__(self, config: LLMConfig, api_key: Optional[str] = None) -> None:
        self._config = config
        self._logger = logger
        
        # Initialize LLM provider
        provider_name = (config.provider or "google").lower()
        provider_cls = LLM_PROVIDERS.get(provider_name)
        
        if provider_cls is None:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        
        self._provider = provider_cls(config, api_key)
        
        # Initialize conversation
        self.conversation = Conversation()
        self.conversation.add_message("system", SYSTEM_PROMPT)
        
        self._logger.debug(f"LLM Client initialized with provider: {provider_name}")
    
    def chat(
        self, 
        user_query: str, 
        retrieved_chunks: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """
        Main chat method - handles everything.
        
        This is the ONLY method needed for all interactions.
        The LLM handles all logic: greetings, searches, refinements, alternatives, etc.
        
        Args:
            user_query: User's question
            retrieved_chunks: Retrieved entities from database (optional)
            
        Returns:
            LLM's response
        """
        try:
            # Add user message to conversation
            self.conversation.add_message("user", user_query)
            
            # Build prompt with context
            prompt = self._build_prompt(user_query, retrieved_chunks)
            
            # Get LLM response
            messages = self.conversation.get_messages()[:-1]  # All except current user query
            messages.append({"role": "user", "content": prompt})
            
            self._logger.debug(f"🔍 LLM: Sending prompt to LLM (length: {len(prompt)} chars)")
            self._logger.debug(f"🔍 LLM: Full prompt being sent:\n{prompt}")
            
            response = self._provider.generate(messages)
            
            self._logger.debug(f"🔍 LLM: LLM response received (length: {len(response)} chars)")
            self._logger.debug(f"🔍 LLM: Full LLM response:\n{response}")
            
            # Add assistant response to conversation
            self.conversation.add_message("assistant", response)
            
            # Track shown entities from response (extract IDs from response)
            if retrieved_chunks:
                self._track_shown_entities(response, retrieved_chunks)
            
            return response
            
        except Exception as exc:
            self._logger.error(f"Chat failed: {exc}")
            return self._get_fallback_response()
    
    def _build_prompt(
        self, 
        user_query: str, 
        retrieved_chunks: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """
        Build prompt with context for LLM.
        
        Structure:
        1. Conversation summary (memory)
        2. Previously shown IDs
        3. Current query
        4. Retrieved entities/properties (if any)
        """
        # Get conversation summary first (this is the memory)
        conversation_summary = self.conversation.get_conversation_summary()
        
        # Get previously shown IDs
        shown_ids = self.conversation.get_shown_entities()
        shown_ids_str = ", ".join(shown_ids) if shown_ids else "None"
        
        # Format retrieved properties
        properties_json = "[]"
        total_count = 0
        
        # ALWAYS include properties if they exist (LLM will decide if they're relevant)
        if retrieved_chunks:
            total_count = len(retrieved_chunks)
            properties_list = []
            
            for chunk in retrieved_chunks[:20]:  # Limit to 20 to avoid context overflow
                # Extract key information from metadata
                meta = chunk.metadata
                properties_list.append({
                    "id": meta.get("id"),
                    "title": meta.get("property_title"),
                    "description": meta.get("property_description", "")[:200],  # Truncate
                    "location": {
                        "emirate": meta.get("emirate"),
                        "city": meta.get("city"),
                        "community": meta.get("community"),
                        "subcommunity": meta.get("subcommunity"),
                        "building": meta.get("building_name")
                    },
                    "details": {
                        "rent": meta.get("rent_charge"),
                        "bedrooms": meta.get("number_of_bedrooms"),
                        "bathrooms": meta.get("number_of_bathrooms"),
                        "size": meta.get("property_size"),
                        "type": meta.get("property_type_name"),
                        "furnishing": meta.get("furnishing_status")
                    },
                    "amenities": self._extract_amenities(meta),
                    "availability": meta.get("available_from"),
                    "score": round(chunk.score, 2)
                })
            
            properties_json = json.dumps(properties_list, indent=2)
        
        # Build final prompt using template
        prompt = CONTEXT_TEMPLATE.format(
            total_count=total_count,
            properties_json=properties_json,
            conversation_history=conversation_summary,
            previously_shown_ids=shown_ids_str,
            current_query=user_query
        )
        
        return prompt
    
    def _extract_amenities(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract amenities from metadata (generic approach)"""
        amenities = []
        
        # Boolean amenities (True values only)
        boolean_checks = {
            "swimming_pool": "Pool",
            "gym_fitness_center": "Gym",
            "parking": "Parking",
            "maids_room": "Maid's Room",
            "security_available": "Security",
            "concierge_available": "Concierge",
            "balcony_terrace": "Balcony",
            "pet_friendly": "Pet-Friendly",
            "beach_access": "Beach Access",
            "smart_home_features": "Smart Home"
        }
        
        for field, label in boolean_checks.items():
            value = metadata.get(field)
            if value is True:
                amenities.append(label)
            elif isinstance(value, list) and len(value) > 0:
                # Handle array amenities (e.g., parking, pools)
                amenities.append(label)
        
        return amenities[:10]  # Limit to top 10
    
    def _track_shown_entities(
        self, 
        response: str, 
        retrieved_chunks: List[RetrievedChunk]
    ) -> None:
        """Track which entities were shown in the response"""
        # Simple heuristic: if entity ID appears in response, it was shown
        for chunk in retrieved_chunks:
            entity_id = chunk.metadata.get("id", "")
            if entity_id and entity_id in response:
                self.conversation.track_shown_entity(entity_id)
    
    def _get_fallback_response(self) -> str:
        """Fallback response if LLM fails"""
        return (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again in a moment, or rephrase your question."
        )
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate response from a single prompt (for utility use cases).
        
        This is used by other components (e.g., filter extraction).
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            return self._provider.generate(messages)
        except Exception as exc:
            self._logger.error(f"Generate failed: {exc}")
            return ""
    
    def extract_requirements(self, conversation_context: str) -> Dict[str, Any]:
        """
        Extract user requirements for saving.
        
        Returns structured requirements from conversation.
        """
        try:
            prompt = f"{REQUIREMENT_COLLECTION_PROMPT}\n\nConversation:\n{conversation_context}"
            response = self.generate(prompt)
            
            # Extract JSON from response
            json_match = response.find("{")
            if json_match != -1:
                json_end = response.rfind("}") + 1
                json_str = response[json_match:json_end]
                return json.loads(json_str)
            
        except Exception as exc:
            self._logger.error(f"Requirement extraction failed: {exc}")
        
        return {
            "requirements": {},
            "contact": {},
            "ready_to_submit": False
        }
    
    def send_requirements_to_api(
        self, 
        conversation_summary: str,
        requirements: Dict[str, Any], 
        contact: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send collected requirements to backend API.
        
        Args:
            conversation_summary: LLM-generated brief summary of the conversation
            requirements: User requirements (location, bedrooms, budget, etc.)
            contact: Contact info (name, email, phone)
        
        Returns:
            API response or error dict
        """
        from .config import get_settings
        import requests
        
        try:
            settings = get_settings()
            
            # Get API endpoint from config
            if not hasattr(settings, 'requirement_gathering'):
                self._logger.error("requirement_gathering not configured")
                return {"success": False, "error": "API endpoint not configured"}
            
            endpoint = settings.requirement_gathering.endpoint
            if not endpoint:
                self._logger.error("requirement_gathering.endpoint not set")
                return {"success": False, "error": "API endpoint not configured"}
            
            # Build requirement summary name (brief title)
            name_parts = []
            if requirements.get("bedrooms"):
                name_parts.append(f"{requirements['bedrooms']} Bed")
            if requirements.get("furnishing_status"):
                name_parts.append(requirements['furnishing_status'].title())
            if requirements.get("property_type"):
                name_parts.append(requirements['property_type'].title())
            if requirements.get("location"):
                name_parts.append(f"in {requirements['location']}")
            
            name = " ".join(name_parts) if name_parts else "Property Search Request"
            
            # Build remark (contact info: phone, email)
            remark_parts = []
            if contact.get("phone"):
                remark_parts.append(contact.get("phone"))
            if contact.get("email"):
                remark_parts.append(contact.get("email"))
            remark = ", ".join(remark_parts)
            
            # Prepare payload for API (EXACTLY matching your format)
            payload = {
                "request": {
                    "name": name,
                    "operator": contact.get("name", "User"),
                    "remark": remark,
                    "details": {
                        "description": conversation_summary  # LLM's brief summary!
                    }
                }
            }
            
            self._logger.debug(f"Sending requirements to API: {endpoint}")
            self._logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Send to API
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            response.raise_for_status()
            
            self._logger.debug(f"Requirements sent successfully: {response.status_code}")
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response": response.json() if response.content else {}
            }
            
        except requests.exceptions.RequestException as exc:
            self._logger.error(f"Failed to send requirements to API: {exc}")
            return {
                "success": False,
                "error": str(exc)
            }
        except Exception as exc:
            self._logger.error(f"Unexpected error sending requirements: {exc}")
            return {
                "success": False,
                "error": str(exc)
            }
    
    def collect_and_send_requirements(self) -> Dict[str, Any]:
        """
        Collect requirements from conversation and send to API.
        LLM creates a brief conversation summary and extracts contact info.
        
        Returns:
            Dict with success status and message
        """
        try:
            # Get conversation context (last 15 messages for full context)
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation.get_messages()[-15:]
            ])
            
            # Let LLM extract requirements and create summary intelligently
            extracted = self.extract_requirements(conversation_text)
            
            self._logger.debug(f"Extracted data: {json.dumps(extracted, indent=2)}")
            
            # Get LLM's conversation summary (this goes to details.description)
            conversation_summary = extracted.get("conversation_summary", "User property search request")
            
            # Send to API with LLM's summary
            result = self.send_requirements_to_api(
                conversation_summary,  # LLM's brief summary (NOT structured JSON!)
                extracted.get("requirements", {}),
                extracted.get("contact", {})
            )
            
            return result
                
        except Exception as exc:
            self._logger.error(f"collect_and_send_requirements failed: {exc}")
            return {
                "success": False,
                "error": str(exc)
            }
    
    def clear_conversation(self) -> None:
        """Clear conversation history"""
        self.conversation.clear()
        self.conversation.add_message("system", SYSTEM_PROMPT)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            "message_count": len(self.conversation.messages),
            "interaction_count": self.conversation.interaction_count,
            "shown_entities_count": len(self.conversation.shown_entities),
            "has_preferences": bool(self.conversation.user_preferences),
            "last_activity": self.conversation.last_activity_time
        }
