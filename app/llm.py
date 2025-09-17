from __future__ import annotations
from typing import Callable, Dict, List, Optional, Type
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

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages


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


def is_property_query(query: str) -> bool:
    query = query.lower().strip()

    # If user asks "what is ..." → it's a definition, not a search
    if query.startswith("what is") or query.startswith("define "):
        return False

    # Keywords that indicate the user is looking for listings
    search_keywords = [
        "show", "find", "rent", "buy", "lease", 
        "apartment", "villa", "flat", "studio", 
        "house", "bedroom", "property in", "listings"
    ]

    return any(word in query for word in search_keywords)


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
            "\n"
            "Greetings:\n"
            "- If the user greets you (hi, hello, hey), greet them back warmly and ask one friendly follow-up question "
            "(e.g., 'Which UAE city are you interested in — Dubai, Abu Dhabi, or somewhere else?').\n"
            "\n"
            "Non-property Queries:\n"
            "- If the user asks about something unrelated to properties (e.g., sports, politics, weather), politely explain "
            "that you can only assist with UAE property-related queries, and guide them back with a property-related question.\n"
            "\n"
            "Outside UAE:\n"
            "- If the user asks about properties outside UAE, politely explain that you only support UAE properties, and ask "
            "a follow-up question about UAE instead.\n"
            "\n"
            "Clarifying Questions:\n"
            "- If the user asks to see properties in a UAE location, do NOT show listings immediately.\n"
            "- Instead, ask one follow-up question at a time until you have enough details:\n"
            "  • Budget or price range\n"
            "  • Property type (apartment, villa, studio, etc.)\n"
            "  • Number of bedrooms/bathrooms\n"
            "  • Location (city, neighborhood)\n"
            "  • Size (sqft or sqm)\n"
            "- Only after gathering sufficient details, you may use retrieved property data to answer.\n"
            "\n"
            "General Knowledge:\n"
            "- If the user asks general questions like 'What is property?' or 'What is an apartment?', provide a clear and "
            "helpful explanation. Do NOT show property listings or data for these questions.\n"
            "\n"
            "Identity:\n"
            "- If the user asks 'Who are you?', respond: 'I am LeaseOasis, your friendly UAE property assistant, here to help "
            "you find and understand leasing options.'\n"
            "\n"
            "Sources:\n"
            "- Only show sources (table and id) when giving factual property details from retrieved chunks.\n"
            "- Do NOT show sources when answering general knowledge, greetings, or clarifying questions.\n"
            "\n"
            "Tone & Experience:\n"
            "- Use simple, clear, and engaging language.\n"
            "- Ask only ONE clarifying question at a time.\n"
            "- Always acknowledge vague queries politely and guide the user naturally.\n"
        )

    def chat(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Process user query with greetings, clarifications, or polite handling of out-of-scope queries."""
        user_input = user_input.strip()

        # Construct follow-up instruction block
        followup_prompt = (
            f"{user_input}\n"
            "Important rules:\n"
            "- If greeting, greet back and ask a friendly clarifying property question.\n"
            "- If unrelated to property, politely explain that I only handle UAE property queries and guide back.\n"
            "- If outside UAE, politely explain I only support UAE properties and guide back.\n"
            "- If about UAE properties, ask one clarifying question at a time (budget, type, bedrooms, size).\n"
            "- If about general property knowledge, explain clearly without showing listings.\n"
            "- If asked who I am, introduce myself as LeaseOasis, the UAE property assistant.\n"
            "- Never list properties unless enough details are collected and retrieved context matches.\n"
        )

        self.conversation.add_message("user", followup_prompt)
        response = self._safe_generate(self.conversation.get_messages())
        self.conversation.add_message("assistant", response)
        return response

    def generate(self, prompt: str) -> str:
        """Fallback for BaseLLM compatibility."""
        return self._safe_generate([{"role": "user", "content": prompt}])

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
