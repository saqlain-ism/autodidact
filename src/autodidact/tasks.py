"""Task and checker primitives shared by every sandbox app.

A ``Task`` is a plain-English request ("Restock every low Dairy product...")
plus two functions:

* ``check(before, after, answer)`` decides pass or fail from the world state
  before and after the episode, and the agent's final answer. It looks at
  *everything* that changed, so doing the right thing plus something unasked
  for is a failure.
* ``reference(session)`` is a scripted solution that uses only the public MCP
  tools. Tests run it for every task to prove each task is solvable and each
  checker accepts a correct solution.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from autodidact.envs.base import Snapshot

if TYPE_CHECKING:
    from autodidact.envs.session import ToolSession


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str

    @classmethod
    def ok(cls) -> CheckResult:
        return cls(True, "ok")

    @classmethod
    def fail(cls, reason: str) -> CheckResult:
        return cls(False, reason)


Checker = Callable[[Snapshot, Snapshot, str | None], CheckResult]
Reference = Callable[["ToolSession"], Awaitable[str | None]]


@dataclass(frozen=True)
class Task:
    id: str
    app: str
    tier: str  # easy | medium | hard
    template: str
    prompt: str
    world_seed: int
    expects_answer: bool
    check: Checker
    reference: Reference


# ------------------------------------------------------------------ answer parsing

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(answer: str | None) -> Decimal | None:
    """First number in an answer, ignoring currency symbols and thousands separators.

    "₹1,24,350.50" -> 124350.50 ; "There are 11 products." -> 11
    """
    if not answer:
        return None
    cleaned = answer.replace(",", "").replace("₹", " ")
    cleaned = re.sub(r"\b(?:rs|inr)\.?", " ", cleaned, flags=re.IGNORECASE)
    match = _NUMBER.search(cleaned)
    if not match:
        return None
    try:
        return Decimal(match.group())
    except InvalidOperation:
        return None


def number_matches(
    answer: str | None, expected: Decimal | int, tolerance: str = "0"
) -> CheckResult:
    got = parse_number(answer)
    if got is None:
        return CheckResult.fail(f"no number in answer {answer!r}")
    if abs(got - Decimal(expected)) > Decimal(tolerance):
        return CheckResult.fail(f"answered {got}, expected {expected}")
    return CheckResult.ok()
