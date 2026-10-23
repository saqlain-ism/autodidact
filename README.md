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
like real internal API docs, and leave out ten deliberate quirks the agent has
to discover by itself:

| Quirk | What goes wrong if you don't know it |
|---|---|
| Money is in paise | Setting a price of ₹249.50 as `249` charges ₹2.49 |
| Cursor pagination, max 20 per page | Counts and totals silently miss products |
| Archived products are hidden by default | "Next free SKU" collides with an archived one |
| Search returns at most 5 results | "Most expensive Kaveri product" picks the wrong one |
| SKU prefix must match the category code | `sku rejected`, with no explanation |
| Stock adjustments need a reason from a hidden list | `reason not accepted` |
| Purchase orders: draft, then submitted, then received | Stock never arrives |
| Suppliers have minimum order quantities | `quantity rejected for one or more lines` |
| The main supplier is not always the cheapest | Overpaying on restocks |
| Sales orders list oldest first | "Most recent order" is on the last page |

Worlds are generated from a seed. The **test world** is reserved for hidden
evaluation; the agent only ever practises on other seeds, so it cannot memorise
answers.

### 60 hidden test tasks

15 templates x 4 instances, in three tiers (20 easy, 20 medium, 20 hard), from
"How many units of STP-0188 are in stock?" to "Restock every low Dairy product
from its cheapest supplier, respecting minimum order quantities, and receive the
deliveries."

Every task has an automatic checker that diffs the whole database before and
after the episode, so doing the right thing *plus* something unasked for fails.
Mistakes the agent fully undoes itself (an order it placed and then cancelled)
are forgiven.

The test suite proves the evaluation is fair:

- a scripted reference solution, using only the MCP tools, passes all 60 tasks;
- doing nothing fails every task that asks for a change;
- a wrong answer fails every question;
- a correct solution plus one unrelated change fails.

### The agent

A plain think, act, observe loop over the MCP tools, ending with a `finish`
call. Its system prompt is generic (it never mentions StockRoom) and identical
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
pytest                          # 187 tests, no network needed
autodidact doctor               # real two-turn tool-calling test per provider
autodidact eval --per-template 1                 # 15 tasks: a quick baseline
autodidact eval                                  # all 60 tasks
autodidact report runs\docs-only__gemini-gemini-3.5-flash-lite
autodidact show runs\docs-only__gemini-gemini-3.5-flash-lite inv-restock_one-1
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
  eval/runner.py             resumable evaluation on the hidden tasks
  eval/report.py             pass rates with confidence intervals
  cli.py                     autodidact doctor | eval | report | show
tests/                       187 tests; providers are simulated, no network
```
