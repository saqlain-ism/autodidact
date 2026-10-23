"""Test doubles: a scripted chat model that needs no network."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from autodidact.llm.base import LLMResponse, ToolCallRequest, Usage

Step = dict[str, Any] | Callable[[list[dict[str, Any]]], dict[str, Any]]


class ScriptedLLM:
    """Plays back a list of steps. A step is ``{"content": ..., "calls": [(name, args), ...]}``
    or a function of the messages it receives (for replies that depend on tool results)."""

    name = "scripted:test"

    def __init__(self, steps: list[Step], repeat_last: bool = False):
        self.steps = list(steps)
        self.repeat_last = repeat_last
        self.requests: list[list[dict[str, Any]]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []

    async def chat(self, messages, tools, *, sample: int = 0) -> LLMResponse:
        self.requests.append(json.loads(json.dumps(messages)))
        self.tools_seen.append(tools)
        if not self.steps:
            raise AssertionError("script exhausted")
        step = self.steps[0] if (self.repeat_last and len(self.steps) == 1) else self.steps.pop(0)
        if callable(step):
            step = step(messages)
        n = len(self.requests)
        calls, raw = [], []
        for i, (name, args) in enumerate(step.get("calls", [])):
            cid = f"c{n}_{i}"
            text = args if isinstance(args, str) else json.dumps(args)
            parsed, error = (
                ({}, "arguments are not valid JSON") if isinstance(args, str) else (args, None)
            )
            calls.append(ToolCallRequest(cid, name, parsed, text, error))
            raw.append(
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": name, "arguments": text},
                    "extra_content": {"google": {"thought_signature": f"sig-{n}-{i}"}},
                }
            )
        message = {"role": "assistant", "content": step.get("content")}
        if raw:
            message["tool_calls"] = raw
        return LLMResponse(
            step.get("content"),
            calls,
            message,
            Usage(prompt_tokens=100, completion_tokens=10, requests=1),
        )


def last_tool_result(messages: list[dict[str, Any]]) -> Any:
    content = next(m["content"] for m in reversed(messages) if m["role"] == "tool")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content
