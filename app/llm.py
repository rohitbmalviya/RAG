from __future__ import annotations

from typing import List

import google.generativeai as genai
try:
    from openai import OpenAI
except Exception:  
    OpenAI = None  

from .config import LLMConfig, get_settings
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
        f"Instructions:\n{instructions}\n\n" \
        f"User question:\n{user_query}\n\n" \
        f"Context:\n{context_block}\n\n" \
        f"Answer:"
    )
    return prompt


class LLMClient:
    def __init__(self, config: LLMConfig, api_key: str | None) -> None:
        self._logger = get_logger(__name__)
        self._config = config
        provider = (config.provider or "google").lower()
        self._provider = provider
        if not config.model:
            self._logger.warning("LLM model is not set; generation may fail")
        if provider == "google":
            if api_key:
                genai.configure(api_key=api_key)
            else:
                self._logger.warning("LLM API key missing for Google provider")
            self._model = genai.GenerativeModel(model_name=config.model)
            self._client_oa = None
        elif provider in {"openai", "azure_openai"}:
            if OpenAI is None:
                raise RuntimeError("openai package not available. Add it to requirements.txt")
            from os import getenv as _getenv
            key = api_key or _getenv("OPENAI_API_KEY")
            if not key:
                self._logger.warning("OPENAI_API_KEY missing for OpenAI provider")
            self._client_oa = OpenAI(api_key=key)  
            self._model = None  
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def generate(self, prompt: str) -> str:
        if self._provider == "google":
            try:
                response = self._model.generate_content(  
                    prompt,
                    generation_config={
                        "temperature": self._config.temperature,
                        "max_output_tokens": self._config.max_output_tokens,
                    },
                )
            except Exception as exc:
                self._logger.error("LLM Google generate_content failed: %s", exc)
                raise
            try:
                return response.text
            except Exception:
                return str(response)
        
        try:
            resp = self._client_oa.chat.completions.create(  
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._config.temperature,
                max_tokens=self._config.max_output_tokens,
            )
        except Exception as exc:
            self._logger.error("LLM OpenAI chat.completions.create failed: %s", exc)
            raise
        return resp.choices[0].message.content or ""
