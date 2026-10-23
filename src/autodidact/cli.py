"""Command line: ``autodidact doctor | eval | report | show``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from autodidact.agent.loop import FINISH_TOOL, Episode
from autodidact.config import env, load_dotenv, mask
from autodidact.envs.inventory.tasks import hidden_test_tasks
from autodidact.eval.report import format_report
from autodidact.eval.runner import Outcome, RunMismatch, evaluate, select_tasks
from autodidact.llm.base import LLMError, QuotaExhausted
from autodidact.llm.providers import PROVIDERS, ConfigError, make_llm, resolve

# ------------------------------------------------------------------- doctor

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


async def _doctor_one(provider: str) -> bool:
    """A real two-turn tool-calling round trip: call a tool, get its result, answer."""
    _, spec, model = resolve(provider)
    print(f"\n{provider}  model={model}  key={mask(env(spec.key_env))}")
    if not env(spec.key_env):
        print(f"  skipped: set {spec.key_env} in .env (get a key at {spec.signup_url})")
        return False
    llm = make_llm(provider, cache_dir=None, max_retries=2)
    messages = [
        {"role": "system", "content": "Use the tools. Call finish with the final answer."},
        {
            "role": "user",
            "content": "Use the add tool to add 17 and 25, then call finish with the result.",
        },
    ]
    try:
        first = await llm.chat(messages, [ADD_TOOL, FINISH_TOOL])
        calls = [c for c in first.tool_calls if c.name == "add"]
        if not calls:
            print(f"  FAIL: expected a call to 'add', got {first.content!r}")
            return False
        print(f"  ok   turn 1: called add({calls[0].arguments})")
        messages.append(first.message)
        for c in first.tool_calls:
            out = (
                str(c.arguments.get("a", 0) + c.arguments.get("b", 0)) if c.name == "add" else "ok"
            )
            messages.append({"role": "tool", "tool_call_id": c.id, "content": out})
        second = await llm.chat(messages, [ADD_TOOL, FINISH_TOOL])
        answer = next(
            (str(c.arguments.get("answer")) for c in second.tool_calls if c.name == "finish"),
            second.content or "",
        )
        if "42" not in answer:
            print(f"  FAIL: turn 2 answered {answer!r}, expected 42")
            return False
        print(f"  ok   turn 2: answered {answer!r}  (multi-turn tool calling works)")
        print(f"  tokens used: {first.usage.total_tokens + second.usage.total_tokens}")
        return True
    except QuotaExhausted as exc:
        print(f"  FAIL: daily quota exhausted. {exc}")
    except LLMError as exc:
        print(f"  FAIL: {exc}")
    finally:
        await llm.aclose()
    return False


def cmd_doctor(args: argparse.Namespace) -> int:
    providers = [args.provider] if args.provider else list(PROVIDERS)
    default = env("AUTODIDACT_PROVIDER", "gemini")
    print(f"default provider: {default}  (.env found: {Path('.env').is_file()})")
    results = {p: asyncio.run(_doctor_one(p)) for p in providers}
    print(
        "\nsummary: "
        + ", ".join(f"{p} {'OK' if ok else 'not working'}" for p, ok in results.items())
    )
    return 0 if results.get(default, any(results.values())) else 1


# --------------------------------------------------------------------- eval


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.-]+", "-", text).strip("-").lower()


def cmd_eval(args: argparse.Namespace) -> int:
    tasks = select_tasks(
        hidden_test_tasks(),
        per_template=args.per_template,
        tiers=set(args.tier) if args.tier else None,
        ids=set(args.task) if args.task else None,
    )
    if not tasks:
        print("no tasks selected")
        return 1
    try:
        llm = make_llm(args.provider, args.model)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 1
    knowledge = Path(args.knowledge).read_text(encoding="utf-8") if args.knowledge else None
    out = Path(args.out or f"runs/{args.condition}__{_slug(llm.name)}")
    attempts = len(tasks) * args.runs
    print(f"condition={args.condition}  model={llm.name}  tasks={len(tasks)}  repeats={args.runs}")
    print(f"results -> {out}   (about {attempts * 8} API requests if nothing is cached yet)\n")

    def progress(rec: dict, episode: Episode) -> None:
        mark = "PASS" if rec["passed"] else "fail"
        extra = "" if rec["passed"] else f"  - {rec['reason'][:90]}"
        print(
            f"  {mark}  {rec['task_id']:<28} run {rec['run']}  turns {rec['turns']:>2}  "
            f"tools {rec['tool_calls']:>2}{extra}",
            flush=True,
        )

    async def run() -> Outcome:
        try:
            return await evaluate(
                llm,
                tasks,
                out,
                condition=args.condition,
                runs=args.runs,
                knowledge=knowledge,
                max_turns=args.max_turns,
                on_result=progress,
            )
        finally:
            await llm.aclose()

    try:
        outcome = asyncio.run(run())
    except RunMismatch as exc:
        print(f"error: {exc}")
        return 1
    print(f"\ncompleted {outcome.completed}, already done {outcome.skipped}")
    if outcome.stopped_early:
        print(f"\nSTOPPED: {outcome.message}")
        print(
            "Progress is saved. Run the same command again after the quota resets "
            "(Gemini resets at midnight Pacific time: 12:30 pm IST, or 1:30 pm IST in winter)."
        )
        return 2
    print()
    print(format_report([out]))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print(format_report(args.runs))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    path = Path(args.run) / "episodes" / f"{args.task}__run{args.repeat}.json"
    if not path.exists():
        print(f"not found: {path}")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    ep = data["episode"]
    print(f"{data['task_id']}  {'PASS' if data['passed'] else 'FAIL'}: {data['reason']}\n")
    print(f"TASK: {ep['prompt']}\n")
    for i, turn in enumerate(ep["turns"], start=1):
        if turn["thought"]:
            print(f"[{i}] {turn['thought'].strip()[:400]}")
        for c in turn["calls"]:
            flag = "" if c["ok"] else "   <-- error"
            print(f"[{i}] {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}){flag}")
            if c["result"]:
                print(f"      -> {c['result'][:300]}")
    print(f"\nANSWER: {ep['answer']!r}   stop: {ep['stop_reason']}")
    return 0


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autodidact", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check API keys with a real tool-calling round trip")
    d.add_argument("--provider", choices=list(PROVIDERS))
    d.set_defaults(func=cmd_doctor)

    e = sub.add_parser("eval", help="run the agent on the hidden test tasks")
    e.add_argument(
        "--condition",
        default="docs-only",
        help="label for what the agent knows (default: docs-only)",
    )
    e.add_argument("--provider", choices=list(PROVIDERS))
    e.add_argument("--model")
    e.add_argument("--runs", type=int, default=1, help="repeat every task N times")
    e.add_argument(
        "--per-template",
        type=int,
        choices=[1, 2, 3, 4],
        help="only the first N tasks of each of the 15 templates",
    )
    e.add_argument("--tier", action="append", choices=["easy", "medium", "hard"])
    e.add_argument("--task", action="append", help="a specific task id (repeatable)")
    e.add_argument("--max-turns", type=int, default=25)
    e.add_argument("--knowledge", help="file with learned knowledge to give the agent")
    e.add_argument("--out", help="run folder (default: runs/<condition>__<model>)")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("report", help="summarise one or more run folders side by side")
    r.add_argument("runs", nargs="+")
    r.set_defaults(func=cmd_report)

    s = sub.add_parser("show", help="print the transcript of one attempt")
    s.add_argument("run")
    s.add_argument("task")
    s.add_argument("--repeat", type=int, default=0)
    s.set_defaults(func=cmd_show)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # Windows consoles and ₹
    logging.getLogger("httpx").setLevel(logging.WARNING)  # no line per HTTP request
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
