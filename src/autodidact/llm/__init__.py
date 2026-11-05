"""Chat models with tool calling, behind one small interface."""

from autodidact.llm.base import (
    LLM,
    LLMError,
    LLMResponse,
    ProviderUnavailable,
    QuotaExhausted,
    StopRun,
    ToolCallRequest,
    Usage,
)

__all__ = [
    "LLM",
    "LLMError",
    "LLMResponse",
    "ProviderUnavailable",
    "QuotaExhausted",
    "StopRun",
    "ToolCallRequest",
    "Usage",
]
