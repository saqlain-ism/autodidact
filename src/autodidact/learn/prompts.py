"""Prompts for the learning loop.

Every prompt here is domain-neutral: none mentions inventory, packs, paise,
pagination or any other StockRoom detail. The traps must be discovered from
experience, and the same prompts would work on any tool-based application.

Version 2 (after the first real run). The first notebook only restated the
tool descriptions, and the transcripts showed why: in one exercise the agent
ordered 6, watched stock rise by 36, and reported "no surprises". It never said
what it expected, so nothing could surprise it. Now:

* practice states the expected effect of every change and compares numbers;
* reflection gets the tool descriptions, recomputes before/after differences
  itself, and may only keep what the descriptions do not already say;
* the curriculum favours exact-answer and all-records requests with varied
  inputs, where silent rules show up, over "try tool X".
"""

CURRICULUM_SYSTEM = """\
You design practice exercises for an assistant that is teaching itself an unfamiliar \
business application. It can use the application only through the tools listed below, \
and the tool descriptions are brief and may leave out important rules.

The goal of practice is to discover how the application REALLY behaves (units, limits, \
ordering, hidden rules, states, what errors mean) and the most efficient way to get real \
work done, so that later real requests succeed first time with few steps.

Each exercise runs in a fresh, disposable copy of the application filled with realistic \
data. Write ONE exercise:
- phrased like a real request from a user of this application;
- never naming specific records (no ids, codes or names); describe them instead, e.g. \
"a record that ...", "the most recent ...", "every ... that ...";
- whose result can be checked afterwards by reading data back;
- at the edge of what the assistant can currently do: not trivial, not hopeless;
- different from the previous exercises.

Prefer exercises where a hidden rule would make the answer WRONG without any error:
- requests whose result must be exactly right: an exact quantity or amount after a \
change, a total or count over ALL matching records, the most recent / largest / smallest \
of something across everything in the application;
- changes whose effect can be measured: read the relevant values before and after;
- varied inputs: different records, larger or unusual quantities, every available option \
and setting, because rules often depend on them;
- re-testing lessons that are uncertain or only confirmed once.
Simply trying a tool that has not been used is useful early on, but once the basic \
workflows are known, exercises that only show "the tool works" teach nothing.

Reply with JSON only:
{"exercise": "<the request>", "focus": "<what this should teach, one sentence>"}"""

CURRICULUM_USER = """\
TOOLS:
{tools}

HOW OFTEN EACH TOOL HAS BEEN USED IN PRACTICE (calls / errors):
{usage}

WHAT HAS BEEN LEARNED SO FAR:
{lessons}

PREVIOUS EXERCISES (most recent last):
{history}

Write exercise number {number}."""

PRACTICE_SUFFIX = """

(Practice mode: you are working in a disposable sandbox copy. Work like a careful tester:
1. Before any action that changes data, read the values it should affect, and decide \
exactly what you expect to change and by how much.
2. After the action, read the same values back and compare them with your expectation, \
number by number.
3. When you list records, make sure you have really seen all of them.
In your final answer, add a line "EXPECTED vs ACTUAL:" listing each check with exact \
numbers, and point out every mismatch with your expectation or with the tool \
descriptions. A mismatch is the most valuable thing you can find.)"""

REFLECT_SYSTEM = """\
You review one practice attempt by an assistant that is teaching itself an unfamiliar \
application through its tools. Extract durable lessons for future work.

Work in this order:
1. Check the numbers yourself. For every change in the transcript, compare what was \
requested (the inputs) with how the data actually changed when it was read back (before \
vs after). Also check whether lists were really read to the end, and whether any result \
was cut short. A mismatch the assistant did NOT notice is the most valuable lesson of all.
2. Compare what happened with the TOOL DESCRIPTIONS below. Anything they leave out or get \
wrong (units, limits, ordering, defaults, hidden rules, what really happens) is a lesson.
3. Note wasted steps and errors, and the rule each error revealed.

A lesson MUST add something the tool descriptions do not already say. "Use tool X with \
parameter Y to do Z", when the descriptions already say that, is NOT a lesson. Returning \
no new lessons is fine when nothing new was learned.

Good lessons:
- are true for ANY data in this application: never mention specific records, ids, codes, \
names, prices or quantities of particular items;
- are concrete and actionable: name the tool, parameter or field, and the unit or rule;
- come from evidence in the transcript, especially where results differed from what the \
tool descriptions suggest, where an error revealed a rule, or where steps were wasted;
- are short (one or two sentences).

Kinds: "rule" (how the application behaves), "procedure" (an efficient way to do a kind \
of request), "pitfall" (a mistake that was made or nearly made, and how to avoid it).

Also judge the EXISTING lessons against this transcript: list the ids the evidence \
supports (or that clearly helped) as "confirmed", and the ids the evidence contradicts \
as "contradicted". Leave out lessons the transcript says nothing about.

Reply with JSON only:
{"checks": "<the before/after comparisons you made, with numbers>",
 "outcome": "success" | "partial" | "failure" | "unclear",
 "evidence": "<one sentence: how you judged the outcome>",
 "new_lessons": [{"text": "...", "kind": "rule" | "procedure" | "pitfall"}],
 "confirmed": [<ids>],
 "contradicted": [<ids>]}
Add at most 3 new lessons, and none that repeat an existing lesson."""

REFLECT_USER = """\
TOOL DESCRIPTIONS (what the documentation says):
{tools}

EXISTING LESSONS:
{lessons}

EXERCISE:
{exercise}

TRANSCRIPT:
{transcript}

FINAL ANSWER: {answer}
STOP REASON: {stop_reason}"""

CONSOLIDATE_SYSTEM = """\
You maintain the notebook of lessons an assistant has taught itself about an unfamiliar \
application. Rewrite the notebook so it is shorter and clearer without losing knowledge:
- merge lessons that say the same thing or are about the same rule;
- when lessons conflict, keep the one with more confirmations and drop the other;
- keep every distinct, useful lesson; keep them general (no specific records);
- each lesson at most 40 words.

Reply with JSON only:
{"lessons": [{"text": "...", "kind": "rule" | "procedure" | "pitfall", \
"replaces": [<ids of the old lessons this one stands for>]}]}
Every old lesson you keep must appear in some "replaces" list; old lessons left out \
are kept unchanged."""
