# Autodidact

**An LLM agent that teaches itself to use software it has never seen.**

Plug in an unfamiliar tool server and Autodidact explores it, invents its own
practice tasks, keeps the routines that verifiably work, and gets better at real
tasks it was never shown, with zero human examples.

> Status: work in progress. Done: the sandbox app, the hidden evaluation set, the
> agent, the evaluation harness and the self-directed learning loop. Next: the
> learned-versus-baseline experiments, a second app for the transfer test, and a
> dashboard.

## Baseline (no practice)

60 hidden tasks, `gemini-3.5-flash-lite`, one attempt each:

| | Pass rate | 95% interval | Avg API requests | Avg tokens |
|---|---|---|---|---|
| StockRoom v2, docs only | 80.0% (48/60) | 68% to 88% | 5.7 | 19,531 |

Easy 20/20, medium 16/20, hard 12/20. Of the 12 failures, 10 come from traps
that practice could teach: case packs (5), pagination (4, including one where
the agent decoded the opaque page cursor and **forged its own**, skipping three
pages) and the search cap (1). Two task types use half of all tokens. So the
experiment measures three things: success, cost per task, and reliability.

## What exists so far

### StockRoom: the unfamiliar software

A small inventory app (products, suppliers, purchase orders, sales orders),
exposed as an **MCP server** with 23 tools. Its tool descriptions are terse,
like real internal API docs, and leave out fourteen deliberate quirks the agent
has to discover by itself.

**Silent traps** give no error at all, just a wrong result. These are what a
capable model with no experience actually fails on:

| Quirk | What goes wrong if you don't know it |
|---|---|
| `update_product_price` takes a bare `price`, in paise | "Set it to ₹78" stores ₹0.78 |
| Purchase orders count case packs, not units | Ordering 13 of a 12-pack item delivers 156 units |
| The category filter takes the code (`DRY`), not the name | Filtering by "Dairy" returns an empty list |
| Lists are paginated (`next_cursor`) | Counts and totals silently miss products |
| Archived products are hidden by default | "Next free SKU" collides with an archived one |
| Search returns at most 5 results | "Most expensive Nimbus product" picks the wrong one |
| Sales orders list oldest first | "Most recent order" is on a later page |
| The main supplier is not always the cheapest | Overpaying on restocks |

**Error traps** cost steps, but the error gives a hint:

| Quirk | What the agent sees |
|---|---|
| Inactive suppliers still appear in quotes; drafts to them are accepted | Submitting fails, and the abandoned draft stays behind unless cancelled |
| SKU prefix must match the category code | `sku rejected` |
| Stock adjustments need a reason from a hidden list | `reason not accepted` |
| Purchase orders: draft, then submitted, then received | `cannot do that in its current state` |
| Minimum order quantities | `quantity rejected for one or more lines` |
| Page size is capped at 20 | `invalid page_size` |

Version 1 of StockRoom had mostly error traps. The baseline agent scored 80%
on it, because it reads errors well and recovers. That left too little room to
measure learning, so version 2 adds the silent traps above. Each one is proven
decisive by a test that plays the naive strategy and fails the task.

Worlds are generated from a seed. The **test world** is reserved for hidden
evaluation; the agent only ever practises on other seeds, so it cannot memorise
answers.

### 60 hidden test tasks

15 templates x 4 instances, in three tiers (20 easy, 20 medium, 20 hard), from
"How many units of STP-0188 are in stock?" to "Restock every low Dairy product
from its cheapest supplier that is still accepting orders, ordering as little as
allowed, and receive the deliveries." Where a template can probe a trap, its
instances are chosen so that they do.

Every task has an automatic checker that diffs the whole database before and
after the episode, so doing the right thing *plus* something unasked for fails.
Mistakes the agent fully undoes itself (an order it placed and then cancelled)
are forgiven.

The test suite proves the evaluation is fair:

- a scripted reference solution, using only the MCP tools, passes all 60 tasks;
- doing nothing fails every task that asks for a change;
- a wrong answer fails every question;
- a correct solution plus one unrelated change fails;
- the naive strategy for each silent trap fails.

### The agent

A plain think, act, observe loop over the MCP tools, plus an exact
`calculate` tool (so arithmetic slips are not mistaken for app knowledge),
ending with a `finish` call. Its system prompt is generic (it never mentions StockRoom) and identical
in every experiment. The only thing that changes between "no practice" and
"after practice" is a block of knowledge the agent wrote for itself, so any
improvement is attributable to learning.

### Self-directed learning (`autodidact learn`)

Each practice episode runs in a fresh practice shop (never the hidden test shop):

1. **Curriculum.** The model proposes an exercise from the tool list, which
   tools are under-explored, what it has learned and what it tried before.
   Exercises describe records ("a product that is running low") instead of
   naming them.
2. **Practice with read-back.** The agent does the exercise using everything it
   has learned so far, then verifies the result by reading data back. That is
   how silent traps surface: "I ordered 9 and stock rose by 54?"
3. **Reflection.** The model writes up to three general lessons and marks
   existing ones as confirmed or contradicted. Lessons contradicted more often
   than confirmed are retired.
4. **Consolidation** every few episodes merges duplicates and resolves
   conflicts, keeping evidence counts.

Guards against cheating and noise:

- every prompt is domain-neutral (none mentions packs, paise or pagination),
  so the traps have to be discovered;
- a **leak filter** rejects any lesson that names a specific record (ids, SKUs,
  or supplier, customer and brand names); lessons must be about the application,
  not one shop's data;
- a **contamination report** gives each practice exercise's word overlap with
  its closest hidden task, and counts near-copies;
- a **naive-memory ablation** gives the agent the same practice experience as
  raw transcripts, with no reflection, to show what reflection itself adds.

The notebook (`knowledge.md`) is the only thing that differs between the
"docs only" and "learned" conditions.

### Built for free API tiers

Free tiers limit requests per minute and per day, so the harness is designed
around them:

- **One client for Groq and Gemini** through their OpenAI-compatible endpoints,
  with MCP schemas reduced to a subset both accept.
- **Pacing** keeps requests under the per-minute limit instead of colliding with it.
- **Short limits** are waited out; **daily limits** stop the run at once.
- **Resumable runs**: each result is saved as soon as it is graded, so running
  the same command after the quota resets continues where it stopped.
- **Response cache**: an identical request is never paid for twice.
- **Context compaction**: older tool results are shortened to save tokens.
- **History safety**: assistant messages are sent back byte-for-byte, so
  Gemini's hidden thought signatures on tool calls survive multi-turn use.

Results come with 95% confidence intervals, because on 60 tasks a few points
of difference can be noise.

## Run it (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env          # then paste your keys into .env
pytest                          # 238 tests, no network needed
autodidact doctor               # real two-turn tool-calling test per provider
autodidact eval --per-template 1                 # 15 tasks: a quick baseline
autodidact eval                                  # all 60 tasks
autodidact learn                                 # 15 practice episodes, ~150 requests
autodidact eval --condition learned --knowledge learn\stockroom-v2__gemini-gemini-3.5-flash-lite\knowledge.md
autodidact eval --condition naive-memory --knowledge learn\stockroom-v2__gemini-gemini-3.5-flash-lite\naive_memory.md
autodidact report runs\docs-only__stockroom-v2__gemini-gemini-3.5-flash-lite
autodidact show runs\docs-only__stockroom-v2__gemini-gemini-3.5-flash-lite inv-restock_one-1
```

StockRoom also runs as a standalone MCP server for any MCP client:
`autodidact-inventory-server --seed 1`.

## Layout

```
src/autodidact/
  envs/base.py               snapshots, diffs, money helpers
  envs/session.py            one episode over MCP
  envs/inventory/world.py    seeded world generator
  envs/inventory/app.py      StockRoom business logic and quirks
  envs/inventory/server.py   MCP server (in-process or stdio)
  envs/inventory/tasks.py    the 60 hidden tasks and their checkers
  tasks.py                   Task, CheckResult, answer parsing
  llm/openai_compat.py       Groq/Gemini client: pacing, retries, quota stops
  llm/schema.py              portable tool schemas
  llm/cache.py               on-disk response cache
  llm/providers.py           provider settings from .env
  agent/loop.py              the agent loop
  agent/calculator.py        exact, sandboxed arithmetic tool
  learn/session.py           curriculum, practice, reflection, consolidation
  learn/knowledge.py         lessons, evidence counts, leak filter
  learn/prompts.py           domain-neutral prompts
  eval/runner.py             resumable evaluation on the hidden tasks
  eval/contamination.py      practice-versus-exam overlap report
  eval/report.py             pass rates with confidence intervals
  cli.py                     autodidact doctor | eval | learn | report | show
tests/                       238 tests; providers are simulated, no network
```
