"""One client for every OpenAI-compatible chat endpoint (Groq, Gemini, others).

Built for free tiers, where limits shape everything:

* **Pacing.** Requests are spaced to stay under the requests-per-minute limit,
  instead of firing and waiting for 429 errors.
* **Short limits** (per minute): wait as long as the provider asks, then retry.
* **Long limits** (per day): raise ``QuotaExhausted`` at once, so the caller can
  save progress and resume tomorrow instead of hammering the API.
* **Caching.** Every successful response is stored; identical requests replay
  from disk.
* **History safety.** The assistant message is kept exactly as received, so
  provider-specific hidden fields (Gemini's thought signatures) survive.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from autodidact.llm.base import (
    LLMError,
    LLMResponse,
    ProviderUnavailable,
    QuotaExhausted,
    ToolCallRequest,
    Usage,
)
from autodidact.llm.cache import ResponseCache, request_key

# Waits longer than this are treated as a long (daily) limit.
LONG_WAIT_SECONDS = 120
DAILY_HINTS = ("per day", "perday", "rpd", "tpd", "daily", "tokens per day", "requests per day")


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        rpm: float = 10,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        timeout: float = 120,
        max_retries: int = 8,
        cache: ResponseCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.provider = provider
        self.model = model
        self.name = f"{provider}:{model}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._max_retries = max_retries
        self._cache = cache
        self._sleep = sleep
        self._last_request = -1e9
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ----------------------------------------------------------------- request

    def build_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self._temperature is not None:
            body["temperature"] = self._temperature
        effort = effort or self._reasoning_effort
        if effort:
            body["reasoning_effort"] = effort
        return body

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        sample: int = 0,
        effort: str | None = None,
    ) -> LLMResponse:
        body = self.build_body(messages, tools, effort)
        key = request_key(body, sample)
        if self._cache is not None and (hit := self._cache.get(key)) is not None:
            response = parse_response(hit)
            response.cached = True
            response.usage.requests = 0
            return response
        data = await self._post(body)
        if self._cache is not None:
            self._cache.put(key, data)
        return parse_response(data)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        last_error = "no attempt made"
        for attempt in range(self._max_retries + 1):
            await self._pace()
            try:
                resp = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                last_error = f"network error: {exc!r}"
                await self._wait_unless_last(attempt, backoff(attempt))
                continue
            if resp.status_code == 200:
                return resp.json()
            text = resp.text[:2000]
            if resp.status_code == 429:
                wait = retry_after_seconds(resp)
                if is_long_limit(text, wait):
                    raise QuotaExhausted(f"{self.name}: daily limit reached. {_short(text)}")
                last_error = f"429 rate limited: {_short(text)}"
                await self._wait_unless_last(
                    attempt, wait if wait is not None else min(2**attempt * 2, 60)
                )
                continue
            if resp.status_code >= 500:
                last_error = f"{resp.status_code} server error: {_short(text)}"
                await self._wait_unless_last(attempt, backoff(attempt))
                continue
            raise LLMError(f"{self.name}: HTTP {resp.status_code}: {_short(text)}")
        raise ProviderUnavailable(
            f"{self.name}: gave up after {self._max_retries + 1} attempts ({last_error})"
        )

    async def _wait_unless_last(self, attempt: int, seconds: float) -> None:
        if attempt < self._max_retries:
            await self._sleep(seconds)

    async def _pace(self) -> None:
        wait = self._last_request + self._min_interval - time.monotonic()
        if wait > 0:
            await self._sleep(wait)
        self._last_request = time.monotonic()


# ------------------------------------------------------------------ parsing


def parse_response(data: dict[str, Any]) -> LLMResponse:
    try:
        message = dict(data["choices"][0]["message"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected response shape: {json.dumps(data)[:500]}") from exc
    message["role"] = "assistant"
    message.pop("reasoning", None)  # Groq echoes reasoning text; it is not part of the history
    calls = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        if not tc.get("id"):
            tc["id"] = f"call_{i}"  # some providers omit ids; tool results must reference one
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or ""
        args, error = _parse_arguments(raw)
        calls.append(
            ToolCallRequest(
                id=tc["id"],
                name=fn.get("name", ""),
                arguments=args,
                raw_arguments=raw if isinstance(raw, str) else json.dumps(raw),
                parse_error=error,
            )
        )
    if not message.get("tool_calls"):
        message.pop("tool_calls", None)
    usage = data.get("usage") or {}
    return LLMResponse(
        content=message.get("content"),
        tool_calls=calls,
        message=message,
        usage=Usage(
            prompt_tokens=usage.get("prompt_tokens") or 0,
            completion_tokens=usage.get("completion_tokens") or 0,
            requests=1,
        ),
    )


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw, dict):
        return raw, None
    if not raw or not str(raw).strip():
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"arguments are not valid JSON ({exc.msg})"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


# ------------------------------------------------------------- rate limits

_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?")


def retry_after_seconds(resp: httpx.Response) -> float | None:
    """How long the provider asked us to wait, from headers or the error body."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    text = resp.text
    # Gemini: "retryDelay": "39s"
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', text)
    if m:
        return float(m.group(1))
    # Groq: "Please try again in 850ms" or "... in 7m12.5s"
    m = re.search(r"try again in ([\d.]+)ms", text)
    if m:
        return float(m.group(1)) / 1000
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?)", text)
    if m and m.group(1):
        d = _DURATION.fullmatch(m.group(1))
        if d:
            h, mnt, s = (float(x) if x else 0.0 for x in d.groups())
            return h * 3600 + mnt * 60 + s
    return None


def backoff(attempt: int) -> float:
    """Waits for overloaded servers: 5, 10, 20, 40, then 60 s; about 5 minutes over 8 retries."""
    return float(min(5 * 2**attempt, 60))


def is_long_limit(text: str, wait: float | None) -> bool:
    lowered = text.lower()
    if any(hint in lowered for hint in DAILY_HINTS):
        return True
    return wait is not None and wait > LONG_WAIT_SECONDS


def _short(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"
