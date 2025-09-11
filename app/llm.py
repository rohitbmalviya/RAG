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
    """Simple keyword-based classifier. Replace with LLM check if needed."""
    property_keywords = [
        "property", "apartment", "villa", "rent", "lease",
        "bedroom", "bathroom", "furnished", "size", "sqft",
        "balcony", "garden", "location", "price", "market", "transport"
    ]
    q = query.lower()
    return any(word in q for word in property_keywords)

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
        self.conversation.add_message(
            "system",
            "You are a conversational property leasing assistant focused on UAE properties. "
            "Do NOT provide any property listings or details. "
            "Always ask one clarifying follow-up question at a time "
            "(budget, location within UAE, bedrooms, or property type). "
            "If the user asks about properties outside UAE, respond politely explaining that only UAE properties are supported. "
            "Keep responses natural and human-like."
        )

    def chat(self, user_input: str, retrieved_chunks: Optional[List[RetrievedChunk]] = None) -> str:
        """Process user query with only follow-up questions, no property details."""
        user_input = user_input.strip()

        # Construct prompt instructing LLM to ask only follow-up question
        followup_prompt = (
            f"{user_input}\n"
            "Important: Only ask one friendly clarifying question at a time (budget, city in UAE, "
            "number of bedrooms, or property type). "
            "Do NOT provide any property details or list any apartments. "
            "If the user asks about a country outside UAE, politely explain that only UAE properties are supported."
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
