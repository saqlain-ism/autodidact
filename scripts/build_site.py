"""Build the data file for the replay site from the committed results.

    python scripts/build_site.py

Writes docs/data.js (loaded by docs/index.html). It is a script rather than a
JSON file so the page also works when opened straight from disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autodidact.envs.inventory.tasks import hidden_test_tasks  # noqa: E402
from autodidact.eval.compare import compare  # noqa: E402

RESULTS = ROOT / "results"
PACK = ["restock_one", "restock_category", "fulfil_or_backorder"]
RESULT_CHARS = 360
THOUGHT_CHARS = 400


def load(name: str) -> list[dict]:
    return [
        json.loads(x)
        for x in (RESULTS / name / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def attempt(name: str, record: dict) -> dict:
    path = RESULTS / name / "episodes" / f"{record['task_id']}__run{record['run']}.json"
    episode = json.loads(path.read_text(encoding="utf-8"))["episode"]
    turns = []
    for t in episode["turns"]:
        turns.append(
            {
                "thought": (t["thought"] or "")[:THOUGHT_CHARS],
                "calls": [
                    {
                        "name": c["name"],
                        "args": c["arguments"],
                        "ok": c["ok"],
                        "result": c["result"][:RESULT_CHARS],
                    }
                    for c in t["calls"]
                ],
            }
        )
    return {
        "passed": record["passed"],
        "reason": record["reason"],
        "answer": record["answer"],
        "stop": record["stop_reason"],
        "tokens": record["prompt_tokens"] + record["completion_tokens"],
        "requests": record["requests"],
        "turns": turns,
    }


def main() -> None:
    before, after = load("docs-only"), load("learned")
    tasks = {t.id: t for t in hidden_test_tasks()}
    after_by_key = {(r["task_id"], r["run"]): r for r in after}
    attempts = []
    for b in before:
        a = after_by_key.get((b["task_id"], b["run"]))
        if a is None:
            continue
        t = tasks[b["task_id"]]
        attempts.append(
            {
                "task": t.id,
                "run": b["run"],
                "template": t.template,
                "tier": t.tier,
                "prompt": t.prompt,
                "before": attempt("docs-only", b),
                "after": attempt("learned", a),
            }
        )
    comparison = compare(before, after, groups={"pack": set(PACK)})
    notebook = (RESULTS / "notebooks" / "learner-v2.md").read_text(encoding="utf-8")
    lessons = [line[2:].strip() for line in notebook.splitlines() if line.startswith("- ")]
    data = {
        "model": before[0]["model"],
        "pack_templates": PACK,
        "summary": {
            "overall": {
                k: comparison["overall"][k]
                for k in ("attempts", "passed_before", "passed_after", "p_value")
            }
            | {
                "fixed": len(comparison["overall"]["fixed"]),
                "broken": len(comparison["overall"]["broken"]),
            },
            "pack": {
                k: comparison["groups"]["pack"][k]
                for k in ("attempts", "passed_before", "passed_after", "p_value")
            }
            | {
                "fixed": len(comparison["groups"]["pack"]["fixed"]),
                "broken": len(comparison["groups"]["pack"]["broken"]),
            },
        },
        "lessons": lessons,
        "attempts": attempts,
    }
    out = ROOT / "docs" / "data.js"
    out.write_text(
        "window.AUTODIDACT = "
        + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(attempts)} paired attempts)")


if __name__ == "__main__":
    main()
