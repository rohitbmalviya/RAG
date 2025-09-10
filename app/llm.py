from __future__ import annotations
from typing import Callable, Dict, List, Optional, Type
from .config import LLMConfig, get_settings
from .core.base import BaseLLM
from .models import RetrievedChunk
from .utils import get_logger

def build_prompt(user_query: str, chunks: List[RetrievedChunk]) -> str:
    context_lines: List[str] = []
    settings = get_settings()
    embed_cols = settings.database.embedding_columns or [c for c in settings.database.columns if c != settings.database.id_column]
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        table = meta.get("table") or ""
        identifier = meta.get("id") or meta.get("source_id") or ""
        header = f"[Source {idx}] table={table} id={identifier}"
        context_lines.append(header)
        for col in embed_cols:
            if col in meta and meta[col] is not None:
                context_lines.append(f"{col}: {meta[col]}")
        if chunk.text:
            context_lines.append(chunk.text)
        context_lines.append("")
    context_block = "\n".join(context_lines)

    instructions = (
        "You are a helpful assistant that answers strictly using the provided context. "
        "If the answer cannot be found in the context, reply that you don't know. "
        "Return a concise, factual answer. At the end, list the sources as 'Sources:' followed by table and id."
    )
    prompt = (
        f"Instructions:\n{instructions}\n\n"
        f"User question:\n{user_query}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Answer:"
    )
    return prompt


# Registry for LLM providers
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
            import os as _os
            self._api_key = _os.getenv("LLM_MODEL_API_KEY") or _os.getenv("OPENAI_API_KEY")

    def _setup_client(self) -> None:
        # Optional for providers
        pass

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


@register_llm_provider("google")
class GoogleLLM(BaseLLMProvider):
    def _setup_client(self) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:
            raise RuntimeError("google-generativeai package not available. Add it to requirements.txt") from exc
        if self._api_key:
            genai.configure(api_key=self._api_key)
        else:
            self._logger.warning("LLM API key missing for Google provider")
        self._model = genai.GenerativeModel(model_name=self._config.model)

    def generate(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self._config.temperature,
                    "max_output_tokens": self._config.max_output_tokens,
                },
            )
        except Exception as exc:
            self._logger.error("Google generate_content failed: %s", exc)
            raise
        try:
            return response.text
        except Exception:
            return str(response)


@register_llm_provider("openai")
@register_llm_provider("azure_openai")
class OpenAILLM(BaseLLMProvider):
    def _setup_client(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package not available. Add it to requirements.txt") from exc
        if not self._api_key:
            self._logger.warning("OPENAI_API_KEY missing for OpenAI provider")
        self._client = OpenAI(api_key=self._api_key)

    def generate(self, prompt: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_output_tokens,
            )
        except Exception as exc:
            self._logger.error("OpenAI chat.completions.create failed: %s", exc)
            raise
        return resp.choices[0].message.content or ""


class LLMClient(BaseLLM):
    def __init__(self, config: LLMConfig, api_key: Optional[str]) -> None:
        self._logger = get_logger(__name__)
        self._config = config
        provider_name = (config.provider or "google").lower()
        provider_cls = LLM_PROVIDERS.get(provider_name)
        if provider_cls is None:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
        if not config.model:
            self._logger.warning("LLM model is not set; generation may fail")
        self._provider_client = provider_cls(config, api_key)

    def generate(self, prompt: str) -> str:
        return self._provider_client.generate(prompt)