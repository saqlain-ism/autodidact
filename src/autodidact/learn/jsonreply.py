"""Ask a chat model for a JSON object, tolerating fences and stray text."""

from __future__ import annotations

import json
import re
from typing import Any

from autodidact.llm.base import LLM, Usage

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [m.group(1) for m in FENCE.finditer(text)] + [text]
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


async def ask_json(
    llm: LLM, system: str, user: str, usage: Usage, *, sample: int = 0, effort: str | None = None
) -> dict[str, Any] | None:
    """One request, and one retry if the reply is not a JSON object."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        response = await llm.chat(messages, [], sample=sample, effort=effort)
        usage.add(response.usage)
        value = extract_json(response.content)
        if value is not None:
            return value
        if attempt == 0:
            messages += [
                {"role": "assistant", "content": response.content or ""},
                {
                    "role": "user",
                    "content": "That was not a valid JSON object. Reply with JSON only.",
                },
            ]
    return None
