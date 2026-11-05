"""The agent loop: think, call tools, read results, repeat, then finish.

The system prompt is generic on purpose. It says nothing about StockRoom, and
it is identical in every experimental condition. The only thing that differs
between "no practice" and "after practice" is the ``knowledge`` block, which is
what the agent taught itself. That makes the comparison fair.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from autodidact.agent.calculator import CALCULATE_TOOL, calculate
from autodidact.envs.session import ToolSession
from autodidact.llm.base import LLM, LLMError, StopRun, Usage
from autodidact.llm.schema import to_openai_tools

SYSTEM_PROMPT = """\
You are an operations assistant. You work inside a business application that you \
can only use through the tools provided; you have no other way to see or change its data.

How to work:
- Do exactly what the user asks. Do not make changes they did not ask for.
- Never guess data you can look up. Verify instead of assuming.
- Read tool errors carefully. They tell you what went wrong; adjust and try again.
- When several tool calls do not depend on each other, make them in the same turn.
- Use `calculate` for any arithmetic instead of working it out in your head.
- When the task is done, call `finish`. If the user asked a question, put only the \
answer in `answer` (for example a number, an SKU or a city). If they asked for changes, \
put a one-line summary."""

KNOWLEDGE_HEADER = "\n\nWhat you have learned about this application from earlier practice:\n"

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Call exactly once, when the task is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer to the user's question, or a one-line summary.",
                }
            },
            "required": ["answer"],
        },
    },
}

MAX_RESULT_CHARS = 6000  # a single tool result shown in full
KEEP_RECENT_RESULTS = 8  # older results are shortened to save tokens
OLD_RESULT_CHARS = 400


@dataclass
class ToolStep:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: str


@dataclass
class Turn:
    thought: str | None
    calls: list[ToolStep] = field(default_factory=list)
    cached: bool = False


@dataclass
class Episode:
    prompt: str
    answer: str | None = None
    stop_reason: str = "running"  # finished | text_reply | max_turns | llm_error
    turns: list[Turn] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    seconds: float = 0.0

    @property
    def tool_calls(self) -> int:
        return sum(len(t.calls) for t in self.turns)

    @property
    def tool_errors(self) -> int:
        return sum(not c.ok for t in self.turns for c in t.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "seconds": round(self.seconds, 2),
            "usage": self.usage.as_dict(),
            "turns": [
                {
                    "thought": t.thought,
                    "cached": t.cached,
                    "calls": [
                        {
                            "name": c.name,
                            "arguments": c.arguments,
                            "ok": c.ok,
                            "result": c.result[:2000],
                        }
                        for c in t.calls
                    ],
                }
                for t in self.turns
            ],
        }


def system_prompt(knowledge: str | None = None) -> str:
    if not knowledge or not knowledge.strip():
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + KNOWLEDGE_HEADER + knowledge.strip()


def compact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shorten old tool results. Assistant messages are never touched."""
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    old = set(tool_positions[:-KEEP_RECENT_RESULTS])
    out = []
    for i, m in enumerate(messages):
        if i in old and len(m.get("content") or "") > OLD_RESULT_CHARS:
            m = {**m, "content": m["content"][:OLD_RESULT_CHARS] + " …[older result shortened]"}
        out.append(m)
    return out


async def run_episode(
    llm: LLM,
    session: ToolSession,
    prompt: str,
    *,
    knowledge: str | None = None,
    max_turns: int = 25,
    sample: int = 0,
) -> Episode:
    """Let the agent work on ``prompt`` until it finishes or runs out of turns.

    ``StopRun`` (daily quota, provider outage) is re-raised so the caller can stop and
    resume later instead of recording a failure; any
    other model failure ends the episode with ``stop_reason="llm_error"``.
    """
    started = time.monotonic()
    episode = Episode(prompt=prompt)
    tools = [*to_openai_tools(await session.tool_specs()), CALCULATE_TOOL, FINISH_TOOL]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(knowledge)},
        {"role": "user", "content": prompt},
    ]
    try:
        for _ in range(max_turns):
            response = await llm.chat(compact(messages), tools, sample=sample)
            episode.usage.add(response.usage)
            messages.append(response.message)
            turn = Turn(thought=response.content, cached=response.cached)
            episode.turns.append(turn)

            if not response.tool_calls:
                episode.answer = response.content
                episode.stop_reason = "text_reply"
                return episode

            for call in response.tool_calls:
                if call.name == "finish":
                    episode.answer = str(call.arguments.get("answer", "")) or response.content
                    episode.stop_reason = "finished"
                    turn.calls.append(ToolStep("finish", call.arguments, True, ""))
                    return episode
                if call.parse_error:
                    step = ToolStep(call.name, {}, False, f"ERROR: {call.parse_error}")
                elif call.name == "calculate":
                    out = calculate(str(call.arguments.get("expression", "")))
                    step = ToolStep("calculate", call.arguments, not out.startswith("ERROR"), out)
                else:
                    result = await session.call(call.name, call.arguments)
                    step = ToolStep(call.name, call.arguments, result.ok, result.text())
                turn.calls.append(step)
                content = step.result
                if len(content) > MAX_RESULT_CHARS:
                    content = content[:MAX_RESULT_CHARS] + " …[truncated]"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        episode.stop_reason = "max_turns"
        return episode
    except StopRun:
        raise
    except LLMError as exc:
        episode.stop_reason = "llm_error"
        episode.error = str(exc)
        return episode
    finally:
        episode.seconds = time.monotonic() - started


def describe(episode: Episode) -> str:
    """Readable transcript, for the terminal and for debugging."""
    lines = [f"TASK: {episode.prompt}"]
    for i, turn in enumerate(episode.turns, start=1):
        if turn.thought:
            lines.append(f"[{i}] thinks: {turn.thought.strip()[:300]}")
        for c in turn.calls:
            args = json.dumps(c.arguments, ensure_ascii=False)
            mark = "" if c.ok else "  <-- error"
            lines.append(f"[{i}] {c.name}({args[:200]}){mark}")
            if c.result:
                lines.append(f"      -> {c.result[:200]}")
    lines.append(f"ANSWER: {episode.answer!r} ({episode.stop_reason})")
    return "\n".join(lines)
