"""Chat models with tool calling, behind one small interface."""

from autodidact.llm.base import LLM, LLMError, LLMResponse, QuotaExhausted, ToolCallRequest, Usage

__all__ = ["LLM", "LLMError", "LLMResponse", "QuotaExhausted", "ToolCallRequest", "Usage"]
