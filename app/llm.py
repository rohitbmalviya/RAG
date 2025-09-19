from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Type
import time
from .config import LLMConfig, get_settings
from .core.base import BaseLLM
from .models import RetrievedChunk
from .utils import get_logger


def format_chunk_metadata(chunk: RetrievedChunk, embed_cols: List[str], idx: int) -> str:
    meta = chunk.metadata or {}
    table = meta.get("table") or ""
    identifier = meta.get("id") or meta.get("source_id") or ""
    lines = [f"[Source {idx}] table={table} id={identifier}"]
    for col in embed_cols:
        if col in meta and meta[col] is not None:
            lines.append(f"{col}: {meta[col]}")
    return "\n".join(lines)


def build_prompt(user_query: str, chunks: List[RetrievedChunk], include_sources: bool = False) -> str:
    """Builds a structured prompt for the LLM using retrieved chunks."""
    settings = get_settings()
    embed_cols = settings.database.embedding_columns or [
        c for c in settings.database.columns if c != settings.database.id_column
    ]
    context_lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        context_lines.append(format_chunk_metadata(chunk, embed_cols, idx))
        if chunk.text:
            context_lines.append(chunk.text)
        context_lines.append("")

    context_block = "\n".join(context_lines)
    instructions = (
        "You are a helpful property leasing assistant.\n"
        "- If the user query is vague, ask one friendly clarifying question at a time "
        "(e.g., location, budget, number of bedrooms).\n"
        "- If sufficient details are provided, answer using retrieved property context ONLY.\n"
        "- Do not fabricate property details if context does not match.\n"
        "- If no relevant properties are found, politely say so and ask the next best clarifying question.\n"
        "- Always keep responses conversational and human-like.\n"
    )
    
    if include_sources:
        instructions += "- At the end of factual property answers, list sources as 'Sources:' followed by table and id."
    else:
        instructions += "- Do not include sources or IDs. Just answer naturally based on the context."

    return f"Instructions:\n{instructions}\n\nUser question:\n{user_query}\n\nContext:\n{context_block}\n\nAnswer:"


def filter_retrieved_chunks(chunks: List[RetrievedChunk], min_score: float = 0.7) -> List[RetrievedChunk]:
    """Filter chunks based on relevance score (if available)."""
    return [c for c in chunks if getattr(c, "score", 1.0) >= min_score]


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
            import os
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


def is_greeting(query: str) -> bool:
    """Check if the query is a greeting"""
    query = query.lower().strip()
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", 
        "good evening", "greetings", "howdy", "what's up"
    ]
    return any(greeting in query for greeting in greetings)

def is_general_knowledge_query(query: str) -> bool:
    """Check if the query is asking for general property knowledge"""
    query = query.lower().strip()
    knowledge_keywords = [
        "what is", "define", "explain", "tell me about", 
        "meaning of", "difference between"
    ]
    return any(keyword in query for keyword in knowledge_keywords)

def is_best_property_query(query: str) -> bool:
    """Check if the query is asking for 'best' properties"""
    query = query.lower().strip()
    best_keywords = ["best", "top", "premium", "featured", "recommended", "highest rated", "top-rated"]
    return any(keyword in query for keyword in best_keywords)

def is_average_price_query(query: str) -> bool:
    """Check if the query is asking for average prices"""
    query = query.lower().strip()
    price_keywords = ["average", "mean", "typical", "usual", "normal"]
    price_indicators = ["price", "rent", "cost", "rate"]
    return any(keyword in query for keyword in price_keywords) and any(indicator in query for indicator in price_indicators)

def is_property_query(query: str) -> bool:
    query = query.lower().strip()

    # If user asks "what is ..." → it's a definition, not a search
    if query.startswith("what is") or query.startswith("define "):
        return False

    # Keywords that indicate the user is looking for listings
    search_keywords = [
        "show", "find", "rent", "buy", "lease", 
        "apartment", "villa", "flat", "studio", 
        "house", "bedroom", "property in", "listings",
        "best", "average", "price", "available"
    ]

    return any(word in query for word in search_keywords)

def is_outside_uae_query(query: str) -> bool:
    """Check if the query is about properties outside UAE"""
    query = query.lower().strip()
    non_uae_locations = [
        "london", "new york", "paris", "tokyo", "singapore", 
        "mumbai", "delhi", "bangalore", "chennai", "kolkata",
        "usa", "uk", "canada", "australia", "germany", "france",
        "italy", "spain", "netherlands", "switzerland"
    ]
    return any(location in query for location in non_uae_locations)

def extract_user_preferences_from_conversation(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract user preferences from conversation history"""
    preferences = {}
    for message in messages:
        if message["role"] == "user":
            content = message["content"].lower()
            # Extract location preferences
            if any(location in content for location in ["dubai", "abu dhabi", "sharjah", "ajman"]):
                if "dubai" in content:
                    preferences["emirate"] = "dubai"
                elif "abu dhabi" in content:
                    preferences["emirate"] = "abu dhabi"
                elif "sharjah" in content:
                    preferences["emirate"] = "sharjah"
            
            # Extract bedroom preferences
            import re
            bed_match = re.search(r"(\d+)\s*bed(room)?s?", content)
            if bed_match:
                preferences["number_of_bedrooms"] = int(bed_match.group(1))
            
            # Extract budget preferences
            budget_match = re.search(r"aed\s*(\d+)(?:k|000)?", content)
            if budget_match:
                budget = int(budget_match.group(1))
                if "k" in content:
                    budget *= 1000
                preferences["rent_charge"] = {"lte": budget}
            
            # Extract furnishing preferences
            if "furnished" in content:
                preferences["furnishing_status"] = "furnished"
            elif "unfurnished" in content:
                preferences["furnishing_status"] = "unfurnished"
            elif "semi-furnished" in content:
                preferences["furnishing_status"] = "semi-furnished"
    
    return preferences


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

        # Updated system instructions
        self.conversation.add_message(
            "system",
            "You are LeaseOasis, a conversational UAE property leasing assistant.\n"
            "\n"
            "Core Behavior:\n"
            "- Always sound polite, respectful, and human-like.\n"
            "- Keep track of conversation history for smoother follow-ups.\n"
            "- Only provide property details when the user has given enough details AND retrieved context matches.\n"
            "- Never fabricate or guess property details.\n"
            "- Only support UAE properties - politely redirect non-UAE queries.\n"
            "\n"
            "Greetings:\n"
            "- If the user greets you (hi, hello, hey), greet them back warmly and ask one friendly follow-up question "
            "(e.g., 'Which UAE city are you interested in — Dubai, Abu Dhabi, or somewhere else?').\n"
            "- Keep sources empty for greetings - no property details should be shown.\n"
            "\n"
            "Non-property Queries:\n"
            "- If the user asks about something unrelated to properties (e.g., sports, politics, weather), politely explain "
            "that you can only assist with UAE property-related queries, and guide them back with a property-related question.\n"
            "- Keep sources empty for non-property queries.\n"
            "\n"
            "Outside UAE:\n"
            "- If the user asks about properties outside UAE, politely explain that you only support UAE properties, and ask "
            "a follow-up question about UAE instead.\n"
            "- Keep sources empty for outside UAE queries.\n"
            "\n"
            "General Knowledge Queries:\n"
            "- If the user asks general questions like 'What is property?', 'What is an apartment?', 'What is holiday home ready?', "
            "provide a clear and helpful explanation using web search if needed.\n"
            "- Do NOT show property listings or data for these questions.\n"
            "- Keep sources empty for general knowledge queries.\n"
            "\n"
            "Best Property Queries:\n"
            "- When user asks for 'best properties', prioritize in this order:\n"
            "  1. Properties with premiumBoostingStatus: 'Active', carouselBoostingStatus: 'Active', and bnb_verification_status: 'verified'\n"
            "  2. Properties with bnb_verification_status: 'verified'\n"
            "  3. Properties with carouselBoostingStatus: 'Active'\n"
            "  4. Properties with premiumBoostingStatus: 'Active'\n"
            "- Show property details with sources for best property queries.\n"
            "\n"
            "Average Price Queries:\n"
            "- When user asks for 'average price' or 'average rent', calculate the average rent_charge from the retrieved context.\n"
            "- Provide the average value but do NOT show individual property details in sources.\n"
            "- Keep sources empty for average price queries.\n"
            "\n"
            "Property Search Queries:\n"
            "- If the user asks to see properties in a UAE location, do NOT show listings immediately.\n"
            "- Instead, ask one follow-up question at a time until you have enough details:\n"
            "  • Budget or price range\n"
            "  • Property type (apartment, villa, studio, etc.)\n"
            "  • Number of bedrooms/bathrooms\n"
            "  • Location (city, neighborhood)\n"
            "  • Size (sqft or sqm)\n"
            "- Only after gathering sufficient details, you may use retrieved property data to answer.\n"
            "- Show property details with sources for property search queries.\n"
            "\n"
            "No Matches Found:\n"
            "- When no properties match user criteria, offer two options:\n"
            "  1. Try alternate searches (nearby locations, different budget, alternative property types)\n"
            "  2. Gather user requirements to send to admin team\n"
            "- Check if alternative locations exist in context before suggesting them.\n"
            "\n"
            "Identity:\n"
            "- If the user asks 'Who are you?', respond: 'I am LeaseOasis, your friendly UAE property assistant, here to help "
            "you find and understand leasing options.'\n"
            "- Keep sources empty for identity queries.\n"
            "\n"
            "Sources:\n"
            "- Only show sources (table and id) when giving factual property details from retrieved chunks.\n"
            "- Do NOT show sources for: greetings, general knowledge, average price calculations, identity queries, or outside UAE queries.\n"
            "- Always show sources for: property search results, best property queries, specific property details.\n"
            "\n"
            "Interactive Conversation:\n"
            "- Remember user preferences from conversation history.\n"
            "- Build on previous exchanges naturally.\n"
            "- Ask follow-up questions based on what user has already shared.\n"
            "- Make the conversation feel like talking to a human property expert.\n"
            "\n"
            "Tone & Experience:\n"
            "- Use simple, clear, and engaging language.\n"
            "- Ask only ONE clarifying question at a time.\n"
            "- Always acknowledge vague queries politely and guide the user naturally.\n"
            "- Be conversational and helpful, not robotic.\n"
        )

    def chat(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Process user query with greetings, clarifications, or polite handling of out-of-scope queries."""
        user_input = user_input.strip()

        # Handle different types of queries
        if is_greeting(user_input):
            return self._handle_greeting(user_input)
        elif is_outside_uae_query(user_input):
            return self._handle_outside_uae_query(user_input)
        elif is_general_knowledge_query(user_input):
            return self._handle_general_knowledge_query(user_input)
        elif is_best_property_query(user_input):
            return self._handle_best_property_query(user_input, retrieved_chunks)
        elif is_average_price_query(user_input):
            return self._handle_average_price_query(user_input, retrieved_chunks)
        elif is_property_query(user_input):
            return self._handle_property_query(user_input, retrieved_chunks)
        else:
            return self._handle_general_query(user_input)

    def _handle_greeting(self, user_input: str) -> str:
        """Handle greeting messages"""
        greetings = [
            "Hello! I'm LeaseOasis, your friendly UAE property assistant. I'm here to help you find the perfect property to lease in the UAE. Which city are you interested in — Dubai, Abu Dhabi, or somewhere else?",
            "Hi there! Welcome to LeaseOasis! I specialize in helping people find great properties to lease in the UAE. What brings you here today? Are you looking for an apartment, villa, or something else?",
            "Hey! Great to meet you! I'm LeaseOasis, your UAE property leasing assistant. I can help you find properties in Dubai, Abu Dhabi, Sharjah, and other UAE cities. What type of property are you looking for?"
        ]
        import random
        response = random.choice(greetings)
        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _handle_outside_uae_query(self, user_input: str) -> str:
        """Handle queries about properties outside UAE"""
        response = (
            "I'm sorry, but I can only assist with properties in the UAE. I specialize in helping you find great leasing options in Dubai, Abu Dhabi, Sharjah, and other UAE cities. "
            "Would you like to explore properties in any of these UAE locations instead?"
        )
        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _handle_general_knowledge_query(self, user_input: str) -> str:
        """Handle general property knowledge queries"""
        # Use web search for general knowledge
        try:
            from .utils import get_logger
            logger = get_logger(__name__)
            # For now, provide a helpful response without web search
            response = (
                "I'd be happy to explain that! However, I'm specifically designed to help you find properties to lease in the UAE. "
                "If you have questions about property types, leasing terms, or UAE-specific property information, I can help with that. "
                "Would you like to search for properties instead, or do you have a specific UAE property question?"
            )
        except Exception:
            response = "I can help you with UAE property-related questions. Would you like to search for properties instead?"
        
        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _handle_best_property_query(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Handle 'best property' queries with proper prioritization"""
        if retrieved_chunks:
            # Build context-aware prompt for best properties
            context_prompt = self._build_best_property_prompt(user_input, retrieved_chunks)
            response = self._safe_generate([{"role": "user", "content": context_prompt}])
        else:
            # No results found - offer alternatives
            preferences = self.conversation.get_preferences()
            response = self._handle_no_results(user_input, preferences)

        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _handle_average_price_query(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Handle 'average price' queries"""
        if retrieved_chunks:
            # Calculate average price from context
            context_prompt = self._build_average_price_prompt(user_input, retrieved_chunks)
            response = self._safe_generate([{"role": "user", "content": context_prompt}])
        else:
            response = (
                "I don't have enough data to calculate the average price for your query. "
                "Could you please specify the location and property type you're interested in?"
            )

        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _handle_property_query(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Handle property search queries"""
        # Extract user preferences from conversation
        preferences = extract_user_preferences_from_conversation(self.conversation.get_messages())
        self.conversation.update_preferences(preferences)

        if retrieved_chunks:
            # Build context-aware prompt
            context_prompt = self._build_context_prompt(user_input, retrieved_chunks)
            response = self._safe_generate([{"role": "user", "content": context_prompt}])
        else:
            # No results found - offer alternatives or gather requirements
            response = self._handle_no_results(user_input, preferences)

        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _handle_general_query(self, user_input: str) -> str:
        """Handle general queries"""
        response = (
            "I'm here to help you find properties to lease in the UAE. I can assist with searching for apartments, villas, and other properties in Dubai, Abu Dhabi, and other UAE cities. "
            "What type of property are you looking for, or would you like me to help you with something specific?"
        )
        self.conversation.add_message("user", user_input)
        self.conversation.add_message("assistant", response)
        return response

    def _build_context_prompt(self, user_input: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build a context-aware prompt for property queries"""
        settings = get_settings()
        embed_cols = settings.database.embedding_columns or [
            c for c in settings.database.columns if c != settings.database.id_column
        ]
        
        context_lines: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(f"[Source {idx}]")
            if chunk.text:
                context_lines.append(chunk.text)
            context_lines.append("")

        context_block = "\n".join(context_lines)
        
        instructions = (
            "You are LeaseOasis, a conversational UAE property leasing assistant.\n"
            "Core Behavior:\n"
            "- Always sound polite, respectful, and human-like.\n"
            "- Use the retrieved property context to answer user queries.\n"
            "- Only provide property details when retrieved context matches the query.\n"
            "- Never fabricate or guess property details.\n"
            "- Always keep responses conversational and helpful.\n"
            "- If no exact matches, suggest alternatives or ask clarifying questions.\n"
            "- Show property details with sources for property search queries.\n"
        )

        return f"Instructions:\n{instructions}\n\nUser question:\n{user_input}\n\nContext:\n{context_block}\n\nAnswer:"

    def _build_best_property_prompt(self, user_input: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build a context-aware prompt for best property queries"""
        context_lines: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(f"[Source {idx}]")
            if chunk.text:
                context_lines.append(chunk.text)
            context_lines.append("")

        context_block = "\n".join(context_lines)
        
        instructions = (
            "You are LeaseOasis, a conversational UAE property leasing assistant.\n"
            "Best Property Query Instructions:\n"
            "- Prioritize properties in this exact order:\n"
            "  1. Properties with premiumBoostingStatus: 'Active', carouselBoostingStatus: 'Active', and bnb_verification_status: 'verified'\n"
            "  2. Properties with bnb_verification_status: 'verified'\n"
            "  3. Properties with carouselBoostingStatus: 'Active'\n"
            "  4. Properties with premiumBoostingStatus: 'Active'\n"
            "- Show the top 3-5 best properties with their details.\n"
            "- Always show sources (table and id) for property details.\n"
            "- Be conversational and explain why these are the 'best' properties.\n"
            "- If no premium/verified properties exist, show the best available options.\n"
        )

        return f"Instructions:\n{instructions}\n\nUser question:\n{user_input}\n\nContext:\n{context_block}\n\nAnswer:"

    def _build_average_price_prompt(self, user_input: str, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build a context-aware prompt for average price queries"""
        context_lines: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(f"[Source {idx}]")
            if chunk.text:
                context_lines.append(chunk.text)
            context_lines.append("")

        context_block = "\n".join(context_lines)
        
        instructions = (
            "You are LeaseOasis, a conversational UAE property leasing assistant.\n"
            "Average Price Query Instructions:\n"
            "- Calculate the average rent_charge from the retrieved context.\n"
            "- Provide the average value in AED.\n"
            "- Do NOT show individual property details or sources.\n"
            "- Be conversational and helpful.\n"
            "- If you can't calculate an average, explain why and ask for clarification.\n"
            "- Example format: 'The average rent for 2-bedroom apartments in Dubai Marina is AED 95,000/year based on the available data.'\n"
        )

        return f"Instructions:\n{instructions}\n\nUser question:\n{user_input}\n\nContext:\n{context_block}\n\nAnswer:"

    def _handle_no_results(self, user_input: str, preferences: Dict[str, Any]) -> str:
        """Handle cases where no properties are found"""
        # Generate alternative suggestions based on preferences
        alternatives = self._generate_alternative_suggestions(preferences)
        
        response = (
            "I couldn't find any properties matching your exact criteria. Let me help you in two ways:\n\n"
            "1. **Try alternate searches**: I can suggest similar properties with slightly different criteria:\n"
        )
        
        if alternatives:
            for alt in alternatives[:3]:  # Show top 3 alternatives
                response += f"   • {alt['suggestion']}\n"
        
        response += (
            "\n2. **Gather your requirements**: I can save your preferences so our team can work with agencies to find matching properties for you.\n\n"
            "Which option would you prefer? Or would you like to adjust your search criteria?"
        )
        return response

    def _generate_alternative_suggestions(self, preferences: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alternative search suggestions based on user preferences"""
        alternatives = []
        
        # Location alternatives
        if "emirate" in preferences:
            current_emirate = preferences["emirate"]
            if current_emirate == "sharjah":
                alternatives.append({
                    "suggestion": "Try Dubai (nearby, more options available)",
                    "type": "location"
                })
            elif current_emirate == "dubai":
                alternatives.append({
                    "suggestion": "Try Abu Dhabi (similar lifestyle options)",
                    "type": "location"
                })
        
        # Budget alternatives
        if "rent_charge" in preferences:
            current_budget = preferences["rent_charge"]
            if isinstance(current_budget, dict) and "lte" in current_budget:
                budget = current_budget["lte"]
                alternatives.append({
                    "suggestion": f"Try budget up to AED {int(budget * 1.2):,} (20% higher)",
                    "type": "budget"
                })
                alternatives.append({
                    "suggestion": f"Try budget up to AED {int(budget * 0.8):,} (20% lower)",
                    "type": "budget"
                })
        
        # Bedroom alternatives
        if "number_of_bedrooms" in preferences:
            bedrooms = preferences["number_of_bedrooms"]
            if bedrooms > 1:
                alternatives.append({
                    "suggestion": f"Try {bedrooms - 1} bedroom properties",
                    "type": "bedrooms"
                })
            alternatives.append({
                "suggestion": f"Try {bedrooms + 1} bedroom properties",
                "type": "bedrooms"
            })
        
        # Furnishing alternatives
        if "furnishing_status" in preferences:
            current_furnishing = preferences["furnishing_status"]
            if current_furnishing == "furnished":
                alternatives.append({
                    "suggestion": "Try semi-furnished properties (more options available)",
                    "type": "furnishing"
                })
            elif current_furnishing == "unfurnished":
                alternatives.append({
                    "suggestion": "Try furnished properties",
                    "type": "furnishing"
                })
        
        return alternatives

    def generate(self, prompt: str) -> str:
        """Fallback for BaseLLM compatibility."""
        return self._safe_generate([{"role": "user", "content": prompt}])

    def gather_requirements(self, user_input: str) -> str:
        """Gather user requirements and send to endpoint"""
        from .requirement_gatherer import RequirementGatherer
        
        preferences = self.conversation.get_preferences()
        gatherer = RequirementGatherer()
        
        result = gatherer.gather_requirements(
            user_query=user_input,
            preferences=preferences,
            conversation_summary=self._summarize_conversation(),
            session_id=str(id(self.conversation))
        )
        
        if result["status"] == "success":
            self.conversation.set_requirement_gathered(True)
            return result["message"]
        else:
            return result["message"]

    def _summarize_conversation(self) -> str:
        """Summarize the conversation for requirement gathering"""
        messages = self.conversation.get_messages()
        if not messages:
            return "No conversation history available."
        
        # Simple summarization - in production, you might want to use LLM for this
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        return f"User discussed: {'; '.join(user_messages[-5:])}"  # Last 5 user messages

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
