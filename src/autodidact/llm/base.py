"""Provider-neutral types for chat models with tool calling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.requests += other.requests

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "requests": self.requests,
        }


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest]
    # The assistant message exactly as the provider sent it. It goes back into
    # the history unchanged: Gemini 3 models attach hidden "thought signatures"
    # to tool calls and reject follow-up requests that drop them.
    message: dict[str, Any]
    usage: Usage = field(default_factory=Usage)
    cached: bool = False


class LLMError(Exception):
    """The provider rejected the request, or kept failing after retries."""


class QuotaExhausted(LLMError):
    """A daily (or similarly long) limit was hit. Save progress and resume later."""


class LLM(Protocol):
    name: str

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, sample: int = 0
    ) -> LLMResponse:
        """One chat completion. ``sample`` separates repeated runs in the cache."""
        ...
