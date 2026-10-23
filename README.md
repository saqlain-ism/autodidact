# Autodidact

**An LLM agent that teaches itself to use software it has never seen.**

Plug in an unfamiliar tool server and Autodidact explores it, invents its own
practice tasks, keeps the routines that verifiably work, and gets better at real
tasks it was never shown, with zero human examples.

> Status: work in progress. The sandbox app and hidden evaluation set are done;
> the agent, curriculum and skill library are next.

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

## Run it

```bash
pip install -e ".[dev]"
pytest                                   # 156 tests
autodidact-inventory-server --seed 1     # StockRoom as a stdio MCP server
```

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
tests/                       app rules, task fairness, stdio server
```
