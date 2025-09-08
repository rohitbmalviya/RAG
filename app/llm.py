from __future__ import annotations

from typing import List

import google.generativeai as genai

from .config import LLMConfig
from .models import RetrievedChunk
from .utils import get_logger


def build_prompt(user_query: str, chunks: List[RetrievedChunk]) -> str:
    context_lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        title = meta.get("property_title") or meta.get("title") or ""
        city = meta.get("city") or ""
        district = meta.get("district") or ""
        table = meta.get("table") or ""
        identifier = meta.get("id") or meta.get("source_id") or ""
        header = f"[Source {idx}] {title} | {city} {district} | table={table} id={identifier}"
        context_lines.append(header)
        context_lines.append(chunk.text)
        context_lines.append("")
    context_block = "\n".join(context_lines)

    instructions = (
        "You are a helpful assistant answering questions about real estate properties. "
        "Use only the provided context to answer the user's question. If the answer cannot be found in the context, say you don't know. "
        "Return a concise, factual answer. At the end, list the sources you used as 'Sources:' followed by table and id."
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
        if api_key:
            genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=config.model)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": self._config.temperature,
                "max_output_tokens": self._config.max_output_tokens,
            },
        )
        try:
            return response.text
        except Exception:
            return str(response)
