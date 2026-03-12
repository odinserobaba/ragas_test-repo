import asyncio
import os
from typing import Sequence

from openai import AsyncOpenAI
from openai import RateLimitError


DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "mistral-small-latest"


def _client() -> AsyncOpenAI:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY env var")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("MISTRAL_BASE_URL", DEFAULT_BASE_URL),
    )


def _build_prompt(question: str, retrieved_contexts: Sequence[str]) -> str:
    ctx = "\n\n---\n\n".join([c.strip() for c in retrieved_contexts if c and c.strip()])
    return (
        "Ты — QA-бот. Отвечай строго по КОНТЕКСТУ ниже. "
        "Если в контексте нет ответа, скажи: \"Не знаю\".\n\n"
        f"КОНТЕКСТ:\n{ctx}\n\n"
        f"ВОПРОС: {question}\n"
        "ОТВЕТ:"
    )


async def answer_question(question: str, retrieved_contexts: Sequence[str]) -> str:
    model = os.getenv("MISTRAL_MODEL", DEFAULT_MODEL)
    max_retries = int(os.getenv("MISTRAL_RATE_LIMIT_RETRIES", "4"))

    messages = [
        {
            "role": "system",
            "content": "Follow the user instructions strictly. Be concise.",
        },
        {
            "role": "user",
            "content": _build_prompt(question=question, retrieved_contexts=retrieved_contexts),
        },
    ]

    for attempt in range(max_retries):
        try:
            resp = await _client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
            )
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = 2 ** (attempt + 1)  # 2, 4, 8, ...
            await asyncio.sleep(delay)
