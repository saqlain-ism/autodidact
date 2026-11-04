"""Prompts for the learning loop.

Every prompt here is domain-neutral: none mentions inventory, packs, paise,
pagination or any other StockRoom detail. The traps must be discovered from
experience, and the same prompts would work on any tool-based application.
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
- aimed at what is least understood: tools never or rarely used, lessons that are still \
uncertain, situations that previously failed, or combinations of tools;
- different from the previous exercises.

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

(Practice mode: you are working in a disposable sandbox copy. Do the request, then \
verify the result by reading back the data you changed or relied on, and check that it \
matches what you intended. In your final answer, report anything that surprised you or \
did not match the tool descriptions.)"""

REFLECT_SYSTEM = """\
You review one practice attempt by an assistant that is teaching itself an unfamiliar \
application through its tools. Extract durable lessons for future work.

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
{"outcome": "success" | "partial" | "failure" | "unclear",
 "evidence": "<one sentence: how you judged the outcome>",
 "new_lessons": [{"text": "...", "kind": "rule" | "procedure" | "pitfall"}],
 "confirmed": [<ids>],
 "contradicted": [<ids>]}
Add at most 3 new lessons, and none that repeat an existing lesson."""

REFLECT_USER = """\
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
