"""Run the agent on hidden test tasks and record everything.

Each run directory holds:

* ``meta.json``: condition, model, and which tasks and repeats were requested;
* ``results.jsonl``: one line per (task, repeat), appended as soon as it finishes;
* ``episodes/``: the full transcript of every attempt.

Because results are appended one by one, a run can stop at any moment (daily
quota, closed laptop) and continue later with the same command: finished
attempts are skipped, and cached model responses replay for free. A run folder
is tied to one model, so results from different models are never mixed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autodidact.agent.loop import Episode, run_episode
from autodidact.envs import version_of
from autodidact.envs.session import ToolSession
from autodidact.llm.base import LLM, QuotaExhausted
from autodidact.tasks import Task


class RunMismatch(Exception):
    """The run folder belongs to a different model or condition."""


@dataclass
class Outcome:
    completed: int
    skipped: int
    stopped_early: bool
    message: str = ""


def select_tasks(
    tasks: Iterable[Task],
    *,
    per_template: int | None = None,
    tiers: set[str] | None = None,
    ids: set[str] | None = None,
) -> list[Task]:
    """Pick a subset. ``per_template=2`` keeps the first 2 of every template (30 tasks)."""
    chosen, seen = [], {}
    for t in tasks:
        if ids and t.id not in ids:
            continue
        if tiers and t.tier not in tiers:
            continue
        seen[t.template] = seen.get(t.template, 0) + 1
        if per_template is not None and seen[t.template] > per_template:
            continue
        chosen.append(t)
    return chosen


def done_keys(out_dir: Path) -> set[tuple[str, int]]:
    path = out_dir / "results.jsonl"
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            keys.add((rec["task_id"], rec["run"]))
    return keys


def _check_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    path = out_dir / "meta.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key in ("environment", "model", "condition"):
            if old.get(key) != meta[key]:
                raise RunMismatch(
                    f"{out_dir} was created for {key}={old.get(key)!r}, "
                    f"not {meta[key]!r}. Use a different --out folder."
                )
        meta = {**old, "updated": meta["created"]}
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def record(
    task: Task, run: int, condition: str, model: str, episode: Episode, passed: bool, reason: str
) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "template": task.template,
        "tier": task.tier,
        "run": run,
        "condition": condition,
        "model": model,
        "passed": passed,
        "reason": reason,
        "answer": episode.answer,
        "stop_reason": episode.stop_reason,
        "error": episode.error,
        "turns": len(episode.turns),
        "tool_calls": episode.tool_calls,
        "tool_errors": episode.tool_errors,
        **episode.usage.as_dict(),
        "seconds": round(episode.seconds, 2),
    }


async def evaluate(
    llm: LLM,
    tasks: list[Task],
    out_dir: str | Path,
    *,
    condition: str,
    runs: int = 1,
    knowledge: str | None = None,
    max_turns: int = 25,
    on_result: Callable[[dict[str, Any], Episode], None] | None = None,
) -> Outcome:
    out_dir = Path(out_dir)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    _check_meta(
        out_dir,
        {
            "environment": ",".join(sorted({version_of(t.app) for t in tasks})),
            "condition": condition,
            "model": llm.name,
            "runs": runs,
            "task_ids": [t.id for t in tasks],
            "max_turns": max_turns,
            "knowledge_chars": len(knowledge or ""),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    if knowledge:
        (out_dir / "knowledge.md").write_text(knowledge, encoding="utf-8")

    finished = done_keys(out_dir)
    completed = skipped = 0
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as results:
        for run in range(runs):
            for task in tasks:
                if (task.id, run) in finished:
                    skipped += 1
                    continue
                # Catch the quota stop inside the session: exceptions that cross the
                # MCP client's task group come out wrapped in an ExceptionGroup.
                quota: QuotaExhausted | None = None
                async with ToolSession(task.world_seed) as session:
                    before = session.app.snapshot()
                    try:
                        episode = await run_episode(
                            llm,
                            session,
                            task.prompt,
                            knowledge=knowledge,
                            max_turns=max_turns,
                            sample=run,
                        )
                    except QuotaExhausted as exc:
                        quota = exc
                    after = session.app.snapshot()
                if quota is not None:
                    return Outcome(completed, skipped, True, str(quota))
                verdict = task.check(before, after, episode.answer)
                rec = record(
                    task, run, condition, llm.name, episode, verdict.passed, verdict.reason
                )
                (out_dir / "episodes" / f"{task.id}__run{run}.json").write_text(
                    json.dumps({**rec, "episode": episode.to_dict()}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                results.write(json.dumps(rec, ensure_ascii=False) + "\n")
                results.flush()
                completed += 1
                if on_result:
                    on_result(rec, episode)
    return Outcome(completed, skipped, False)
