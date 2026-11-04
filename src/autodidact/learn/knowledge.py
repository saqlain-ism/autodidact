"""What the agent has taught itself: lessons, their evidence, and how they are shown.

A lesson is one short, general statement about how the application behaves or
how to work with it ("Purchase-order qty counts packs; units = qty x pack_size").
Each lesson keeps score:

* ``confirmed``: later practice episodes where the lesson held or helped;
* ``contradicted``: episodes where the evidence went against it.

Lessons that keep being contradicted are retired, so a wrong early guess does
not poison everything after it.

Lessons must be **general**. Practice happens in shops with different data
from the hidden exam, so a lesson like "SUP-03 is the cheapest supplier" is
useless at best and misleading at worst. ``data_specific`` rejects lessons that
name particular records: IDs, SKUs, or names of suppliers, customers and brands.
That also guarantees nothing from any specific shop's data can leak into the
knowledge the agent takes into the exam.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from autodidact.envs.inventory.tasks import NEW_CUSTOMERS, NEW_SUPPLIERS
from autodidact.envs.inventory.world import BRANDS, CUSTOMERS, SUPPLIER_NAMES

KINDS = ("rule", "procedure", "pitfall")
MAX_LESSON_CHARS = 300
MAX_ACTIVE = 24

ID_PATTERN = re.compile(r"\b(?:[A-Z]{3}-\d{3,}|SUP-\d+|PO-\d+|SO-\d+)\b")
NAMES = sorted(
    {*SUPPLIER_NAMES, *CUSTOMERS, *NEW_CUSTOMERS, *BRANDS, *(s[0] for s in NEW_SUPPLIERS)},
    key=len,
    reverse=True,
)
NAME_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in NAMES) + r")\b", re.IGNORECASE)


def data_specific(text: str) -> str | None:
    """Why a lesson is about particular records rather than the application, or None."""
    if m := ID_PATTERN.search(text):
        return f"mentions a record id ({m.group()})"
    if m := NAME_PATTERN.search(text):
        return f"mentions a specific name ({m.group()})"
    return None


@dataclass
class Lesson:
    id: int
    text: str
    kind: str = "rule"
    confirmed: int = 0
    contradicted: int = 0
    sources: list[int] = field(default_factory=list)  # practice episodes it came from
    active: bool = True
    retired_reason: str | None = None

    @property
    def score(self) -> int:
        return 1 + self.confirmed - 2 * self.contradicted

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lesson:
        return cls(**d)


@dataclass
class KnowledgeBase:
    lessons: list[Lesson] = field(default_factory=list)
    next_id: int = 1
    rejected: list[dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ edits

    def active(self) -> list[Lesson]:
        return [x for x in self.lessons if x.active]

    def get(self, lesson_id: int) -> Lesson | None:
        return next((x for x in self.lessons if x.id == lesson_id and x.active), None)

    def add(self, text: str, kind: str, episode: int) -> Lesson | None:
        text = " ".join(str(text).split())
        if not text:
            return None
        if len(text) > MAX_LESSON_CHARS:
            text = text[: MAX_LESSON_CHARS - 1] + "…"
        if reason := data_specific(text):
            self.rejected.append({"text": text, "reason": reason, "episode": str(episode)})
            return None
        if any(_same(text, x.text) for x in self.active()):
            return None
        lesson = Lesson(
            id=self.next_id, text=text, kind=kind if kind in KINDS else "rule", sources=[episode]
        )
        self.next_id += 1
        self.lessons.append(lesson)
        return lesson

    def judge(self, confirmed: list[int], contradicted: list[int]) -> list[Lesson]:
        """Record evidence from one episode. Returns lessons retired as a result."""
        for i in set(confirmed) - set(contradicted):
            if lesson := self.get(i):
                lesson.confirmed += 1
        retired = []
        for i in set(contradicted):
            if lesson := self.get(i):
                lesson.contradicted += 1
                if lesson.contradicted >= 2 and lesson.contradicted > lesson.confirmed:
                    lesson.active = False
                    lesson.retired_reason = "contradicted by practice"
                    retired.append(lesson)
        return retired

    def replace_active(self, merged: list[dict[str, Any]]) -> None:
        """Apply a consolidation: ``merged`` items carry text, kind and the ids they replace.

        Evidence counts and sources of the replaced lessons are summed, so
        consolidation never erases the record of what was observed.
        """
        old = {x.id: x for x in self.active()}
        used: set[int] = set()
        fresh: list[Lesson] = []
        for item in merged:
            ids = [i for i in item.get("replaces", []) if i in old and i not in used]
            if not ids:
                continue  # consolidation may only rewrite what exists
            text = " ".join(str(item.get("text", "")).split())[:MAX_LESSON_CHARS]
            if not text or data_specific(text):
                continue
            used.update(ids)
            parts = [old[i] for i in ids]
            fresh.append(
                Lesson(
                    id=self.next_id,
                    text=text,
                    kind=item.get("kind") if item.get("kind") in KINDS else parts[0].kind,
                    confirmed=sum(p.confirmed for p in parts) + len(parts) - 1,
                    contradicted=sum(p.contradicted for p in parts),
                    sources=sorted({s for p in parts for s in p.sources}),
                )
            )
            self.next_id += 1
        for lesson in old.values():
            if lesson.id in used:
                lesson.active = False
                lesson.retired_reason = "merged during consolidation"
        unmentioned = [x for x in old.values() if x.id not in used]
        self.lessons = [x for x in self.lessons if not x.active] + unmentioned + fresh
        self._cap()

    def _cap(self) -> None:
        ranked = sorted(self.active(), key=lambda x: (-x.score, x.id))
        for lesson in ranked[MAX_ACTIVE:]:
            lesson.active = False
            lesson.retired_reason = "least supported when over capacity"

    # ---------------------------------------------------------------- display

    def numbered(self) -> str:
        """Lessons with ids, for prompts that judge or merge them."""
        if not self.active():
            return "(none yet)"
        return "\n".join(
            f"[{x.id}] ({x.kind}; confirmed {x.confirmed}, contradicted {x.contradicted}) {x.text}"
            for x in self.active()
        )

    def render(self) -> str:
        """The knowledge block given to the agent: best-supported first, grouped by kind."""
        titles = {
            "rule": "How the application behaves",
            "procedure": "How to do things",
            "pitfall": "Mistakes to avoid",
        }
        out = []
        for kind in KINDS:
            items = sorted(
                (x for x in self.active() if x.kind == kind), key=lambda x: (-x.score, x.id)
            )
            if items:
                out.append(f"{titles[kind]}:")
                out.extend(f"- {x.text}" for x in items)
                out.append("")
        return "\n".join(out).strip()

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "lessons": [x.to_dict() for x in self.lessons],
            "next_id": self.next_id,
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeBase:
        return cls(
            lessons=[Lesson.from_dict(x) for x in d.get("lessons", [])],
            next_id=d.get("next_id", 1),
            rejected=d.get("rejected", []),
        )


def _same(a: str, b: str) -> bool:
    """Cheap duplicate check: near-identical word sets."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.85
