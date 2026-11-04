"""The learning loop: propose an exercise, practise, reflect, keep what is true.

One episode:

1. **Curriculum.** The model proposes an exercise from the tool list, which
   tools are under-explored, what it has learned and what it tried before.
2. **Practice.** The agent does the exercise in a fresh practice shop (never the
   hidden test shop), with everything it has learned so far, and is asked to
   verify the result by reading data back. That read-back is how silent traps
   surface: "I ordered 9 and stock went up by 54?"
3. **Reflection.** The model reviews the transcript, writes up to three new
   general lessons, and marks existing lessons as confirmed or contradicted.
4. Every few episodes the notebook is **consolidated**: duplicates merged,
   conflicts resolved, evidence counts kept.

Everything is saved after each episode. A daily-quota stop loses at most the
episode in progress, and re-running the same command resumes; cached model
replies make the replay free.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autodidact.agent.loop import Episode, run_episode
from autodidact.envs.inventory.world import TEST_WORLD_SEED
from autodidact.envs.session import ToolSession
from autodidact.learn import prompts
from autodidact.learn.jsonreply import ask_json
from autodidact.learn.knowledge import KnowledgeBase
from autodidact.llm.base import LLM, QuotaExhausted, Usage

PRACTICE_SEED_BASE = 1000
HISTORY_IN_PROMPT = 12
TRANSCRIPT_CHARS = 14000
NAIVE_MEMORY_CHARS = 8000


@dataclass
class LearnOutcome:
    episodes_done: int
    stopped: str  # done | budget | quota
    requests: int
    message: str = ""


@dataclass
class State:
    episodes_done: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_usage: dict[str, list[int]] = field(default_factory=dict)  # name -> [calls, errors]
    requests: int = 0
    tokens: int = 0
    consolidate_pending: bool = False
    model: str = ""
    kb: KnowledgeBase = field(default_factory=KnowledgeBase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_done": self.episodes_done,
            "history": self.history,
            "tool_usage": self.tool_usage,
            "requests": self.requests,
            "tokens": self.tokens,
            "consolidate_pending": self.consolidate_pending,
            "model": self.model,
            "knowledge": self.kb.to_dict(),
        }

    @classmethod
    def load(cls, path: Path) -> State:
        if not path.exists():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            episodes_done=d["episodes_done"],
            history=d["history"],
            tool_usage=d["tool_usage"],
            requests=d["requests"],
            tokens=d["tokens"],
            consolidate_pending=d.get("consolidate_pending", False),
            model=d.get("model", ""),
            kb=KnowledgeBase.from_dict(d["knowledge"]),
        )


def practice_seed(episode: int) -> int:
    seed = PRACTICE_SEED_BASE + episode
    if seed == TEST_WORLD_SEED:
        raise ValueError("practice must never use the hidden test world")
    return seed


# ------------------------------------------------------------------ formatting


def tools_text(specs: list[dict[str, Any]]) -> str:
    lines = []
    for s in specs:
        params = ", ".join((s.get("input_schema") or {}).get("properties", {}))
        lines.append(f"- {s['name']}({params}): {s.get('description', '').strip()}")
    return "\n".join(lines)


def usage_text(specs: list[dict[str, Any]], usage: dict[str, list[int]]) -> str:
    return "\n".join(
        f"- {s['name']}: {usage.get(s['name'], [0, 0])[0]} / {usage.get(s['name'], [0, 0])[1]}"
        for s in specs
    )


def history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(none yet: start by getting to know the application)"
    return "\n".join(
        f"{h['episode'] + 1}. [{h['outcome']}] {h['exercise']}"
        for h in history[-HISTORY_IN_PROMPT:]
    )


def transcript_text(episode: Episode) -> str:
    lines = []
    for i, turn in enumerate(episode.turns, start=1):
        if turn.thought:
            lines.append(f"[{i}] thinks: {turn.thought.strip()[:300]}")
        for c in turn.calls:
            if c.name == "finish":
                continue
            args = json.dumps(c.arguments, ensure_ascii=False)[:300]
            lines.append(f"[{i}] {c.name}({args})")
            lines.append(f"    -> {c.result[:500]}")
    text = "\n".join(lines)
    if len(text) > TRANSCRIPT_CHARS:
        half = TRANSCRIPT_CHARS // 2
        text = text[:half] + "\n…[middle of transcript omitted]…\n" + text[-half:]
    return text


def naive_trace(episode: Episode) -> str:
    """Raw experience with no reflection: the naive-memory ablation."""
    calls = []
    for turn in episode.turns:
        for c in turn.calls:
            if c.name == "finish":
                continue
            outcome = "ok" if c.ok else c.result[:120]
            calls.append(
                f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)[:120]}) -> {outcome}"
            )
    return (
        f"Task: {episode.prompt.split(prompts.PRACTICE_SUFFIX)[0]}\n"
        + "\n".join(f"  {x}" for x in calls)
        + f"\n  Result: {(episode.answer or '')[:300]}"
    )


def render_naive(traces: list[str]) -> str:
    """Most recent experience first, cut to a fixed budget."""
    out, size = [], 0
    for trace in reversed(traces):
        if size + len(trace) > NAIVE_MEMORY_CHARS:
            break
        out.append(trace)
        size += len(trace) + 2
    return "Records of earlier attempts in this application:\n\n" + "\n\n".join(out)


# ------------------------------------------------------------------- the loop


async def _consolidate(llm: LLM, kb: KnowledgeBase, usage: Usage) -> bool:
    if len(kb.active()) < 4:
        return True
    reply = await ask_json(llm, prompts.CONSOLIDATE_SYSTEM, "LESSONS:\n" + kb.numbered(), usage)
    if reply and isinstance(reply.get("lessons"), list):
        kb.replace_active([x for x in reply["lessons"] if isinstance(x, dict)])
    return True


async def learn(
    llm: LLM,
    out_dir: str | Path,
    *,
    episodes: int = 15,
    max_requests: int = 200,
    max_turns: int = 20,
    consolidate_every: int = 4,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> LearnOutcome:
    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = State.load(state_path)
    if state.model and state.model != llm.name:
        raise ValueError(f"{out} belongs to model {state.model}, not {llm.name}")
    state.model = llm.name
    emit = on_event or (lambda e: None)
    spent = Usage()

    def save() -> None:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(state_path)
        (out / "knowledge.md").write_text(state.kb.render() + "\n", encoding="utf-8")
        traces = [h["naive_trace"] for h in state.history if h.get("naive_trace")]
        (out / "naive_memory.md").write_text(render_naive(traces) + "\n", encoding="utf-8")

    def stop(kind: str, message: str = "") -> LearnOutcome:
        save()
        return LearnOutcome(state.episodes_done, kind, spent.requests, message)

    try:
        if state.consolidate_pending:
            await _consolidate(llm, state.kb, spent)
            state.consolidate_pending = False
            save()
    except QuotaExhausted as exc:
        return stop("quota", str(exc))

    while state.episodes_done < episodes:
        if spent.requests >= max_requests:
            return stop("budget", f"used {spent.requests} requests in this session")
        i = state.episodes_done
        episode_usage = Usage()
        quota: QuotaExhausted | None = None
        exercise = focus = ""
        episode: Episode | None = None
        reflection: dict[str, Any] | None = None

        # Catch the quota stop inside the session: exceptions that cross the MCP
        # client's task group come out wrapped in an ExceptionGroup.
        async with ToolSession(practice_seed(i)) as session:
            try:
                specs = await session.tool_specs()
                proposal = await ask_json(
                    llm,
                    prompts.CURRICULUM_SYSTEM,
                    prompts.CURRICULUM_USER.format(
                        tools=tools_text(specs),
                        usage=usage_text(specs, state.tool_usage),
                        lessons=state.kb.numbered(),
                        history=history_text(state.history),
                        number=i + 1,
                    ),
                    episode_usage,
                )
                exercise = str((proposal or {}).get("exercise") or "").strip()
                focus = str((proposal or {}).get("focus") or "").strip()
                if not exercise:
                    exercise = (
                        "Look around the application and find out how its main features work."
                    )
                emit({"type": "exercise", "episode": i, "exercise": exercise, "focus": focus})
                episode = await run_episode(
                    llm,
                    session,
                    exercise + prompts.PRACTICE_SUFFIX,
                    knowledge=state.kb.render(),
                    max_turns=max_turns,
                )
                episode_usage.add(episode.usage)
                reflection = await ask_json(
                    llm,
                    prompts.REFLECT_SYSTEM,
                    prompts.REFLECT_USER.format(
                        lessons=state.kb.numbered(),
                        exercise=exercise,
                        transcript=transcript_text(episode),
                        answer=(episode.answer or "")[:1000],
                        stop_reason=episode.stop_reason,
                    ),
                    episode_usage,
                )
            except QuotaExhausted as exc:
                quota = exc
        spent.add(episode_usage)
        state.requests += episode_usage.requests
        state.tokens += episode_usage.total_tokens
        if quota is not None or episode is None:
            return stop("quota", str(quota))

        # ---- apply what this episode taught
        reflection = reflection or {}
        added, rejected_before = [], len(state.kb.rejected)
        for item in reflection.get("new_lessons") or []:
            if isinstance(item, dict) and (
                lesson := state.kb.add(item.get("text", ""), item.get("kind", "rule"), i)
            ):
                added.append(lesson.text)
        retired = state.kb.judge(
            [x for x in reflection.get("confirmed") or [] if isinstance(x, int)],
            [x for x in reflection.get("contradicted") or [] if isinstance(x, int)],
        )
        for turn in episode.turns:
            for c in turn.calls:
                if c.name in ("finish", "calculate"):
                    continue
                counts = state.tool_usage.setdefault(c.name, [0, 0])
                counts[0] += 1
                counts[1] += 0 if c.ok else 1
        outcome = str(reflection.get("outcome") or "unclear")
        state.history.append(
            {
                "episode": i,
                "seed": practice_seed(i),
                "exercise": exercise,
                "focus": focus,
                "outcome": outcome,
                "evidence": reflection.get("evidence", ""),
                "lessons_added": added,
                "requests": episode_usage.requests,
                "naive_trace": naive_trace(episode),
            }
        )
        (out / "episodes" / f"ep{i:03d}.json").write_text(
            json.dumps(
                {
                    "exercise": exercise,
                    "focus": focus,
                    "reflection": reflection,
                    "episode": episode.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        state.episodes_done += 1
        state.consolidate_pending = state.episodes_done % consolidate_every == 0
        save()
        emit(
            {
                "type": "episode",
                "episode": i,
                "outcome": outcome,
                "added": added,
                "rejected": len(state.kb.rejected) - rejected_before,
                "retired": [x.text for x in retired],
                "lessons": len(state.kb.active()),
                "requests": spent.requests,
            }
        )
        if state.consolidate_pending:
            before = len(state.kb.active())
            try:
                await _consolidate(llm, state.kb, spent)
            except QuotaExhausted as exc:
                return stop("quota", str(exc))
            state.consolidate_pending = False
            save()
            emit({"type": "consolidated", "before": before, "after": len(state.kb.active())})

    return stop("done")
