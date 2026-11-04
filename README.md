# Autodidact

**An LLM agent that teaches itself to use software it has never seen.**

Plug in an unfamiliar tool server and Autodidact explores it, invents its own
practice tasks, keeps the routines that verifiably work, and gets better at real
tasks it was never shown, with zero human examples.

> Status: work in progress. Done: the sandbox app, the hidden evaluation set, the
> agent and the evaluation harness. Next: the self-generated curriculum and the
> skill library.

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
pytest                          # 220 tests, no network needed
autodidact doctor               # real two-turn tool-calling test per provider
autodidact eval --per-template 1                 # 15 tasks: a quick baseline
autodidact eval                                  # all 60 tasks
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
  eval/runner.py             resumable evaluation on the hidden tasks
  eval/report.py             pass rates with confidence intervals
  cli.py                     autodidact doctor | eval | report | show
tests/                       220 tests; providers are simulated, no network
```
