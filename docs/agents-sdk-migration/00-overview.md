# Option A: OpenAI SDK → Agents SDK migration — Overview

## What this document set is

This is a precise, complete specification for converting this project's two
raw OpenAI API calls to use the **OpenAI Agents SDK** (`pip install
openai-agents`, imported as `agents`), without changing anything about what
the app actually does. It is written so a beginner can follow it, and so a
future Claude Code session (or you) can implement it from these files alone,
without needing to re-read this whole conversation.

Read the files in this folder in order:

1. `00-overview.md` — this file. The big picture and the ground rules.
2. `01-config-module.md` — changes to `lib/yahoo_finance/config.py`.
3. `02-intent-module.md` — changes to `lib/yahoo_finance/intent.py`.
4. `03-answer-module.md` — changes to `lib/yahoo_finance/answer.py`.
5. `04-testing-and-rollout.md` — the order to make changes in, and how to
   verify each step before moving to the next.
6. `05-non-goals-and-risks.md` — what this migration deliberately does NOT
   do, and the specific things that could go wrong.

## The one-sentence summary

Two Python functions currently build a prompt string by hand, call
`client.responses.create(...)`, and manually parse the text that comes back.
This migration replaces *only the inside* of those two functions with the
Agents SDK's `Agent` + `Runner.run_sync(...)` pattern. Every other file in
the project — the actual business logic — is untouched.

## Why "Option A" and not some other approach

Earlier in this project's design discussion, two migration shapes were
considered:

- **Option A (this one): conservative swap.** Keep the existing pipeline
  (detect intent → fetch data in plain Python → generate answer) exactly as
  it is. Only change *how* the two LLM calls are made internally.
- **Option B: idiomatic single agent.** Let one agent call Yahoo Finance
  fetching as a "tool" itself, inside a multi-turn loop, merging intent
  detection and answer writing into one call.

Option A was chosen because it has a much smaller blast radius: it changes
nothing about what data flows through the app, in what order, or in what
shape. Option B is explicitly **out of scope** for this spec — see
`05-non-goals-and-risks.md`.

## The rule everything else follows from

Two functions in this codebase have a **contract** — a promise about their
input and output types — that other files depend on and that this migration
must not break:

```python
def detect_intent(question: str) -> dict:
    ...

def generate_answer(question: str, intent: dict, yahoo_data) -> str:
    ...
```

`lib/yahoo_finance/pipeline.py` calls both of these and does things like
`intent.get("module")`, `intent["module"] = modules`, and
`intent.get("reason", "")` on whatever `detect_intent()` returns. If
`detect_intent()` ever returned something that *isn't* a plain Python
`dict` (for example, a Pydantic model object), those calls would break
immediately.

**Every code change in this spec exists to preserve those two function
signatures exactly, while changing their internals.** If you're ever unsure
whether a change is "safe," ask: *does the calling code outside this
function still get the same type back?* If yes, you're inside the lines.

## A permissions note, read before you start

This repository's `.claude/CLAUDE.md` marks `lib/` (and `public/`) as
off-limits for Claude to write to directly. `lib/yahoo_finance/config.py`,
`intent.py`, and `answer.py` — the three files this spec changes — are all
inside `lib/`. That means:

- Claude can **read** these files and **write this spec** (which lives in
  `docs/`, outside `lib/`), but cannot apply the changes to `lib/` itself.
- **You** (or a Claude session explicitly told the restriction doesn't apply
  this time) need to be the one who actually copies the "full new file
  content" from `01-config-module.md`, `02-intent-module.md`, and
  `03-answer-module.md` into the real files.
- `requirements.txt` is **not** inside `lib/` or `public/`, so that edit
  (adding the `openai-agents` package) is one Claude can make directly, once
  you're ready to start.

## The diagram

```
 User question
      |
      v
+-----------------------------+
|  AGENT 1: Intent Router     |   was: detect_intent()
|  (1 LLM call)                |
|  in:  the question           |
|  out: Intent object          |   guaranteed shape:
|  {ticker, module,            |   { ticker: [...], module: [...],
|   submodule, parameters,     |     submodule: [...], parameters: {...},
|   reason}                    |     reason: "..." }
+--------------+----------------+
               |  .model_dump() turns it back into
               |  a plain dict before it leaves detect_intent()
               v
+-----------------------------+
|  plain Python, no LLM       |
|  pipeline.py + fetch.py     |   unchanged: force-add "info" for
|  (unchanged)                 |   valuation questions, execute_yahoo_intent(),
|                               |   reorder_statement_data(),
|                               |   compute_valuation_metrics()
+--------------+----------------+
               |  yahoo_data dict
               v
+-----------------------------+
|  AGENT 2: Equity Research    |   was: generate_answer()
|  Analyst (1 LLM call)        |
|  in:  question + intent      |
|       + yahoo_data            |
|  out: plain Markdown text    |   (no output_type set, so
+--------------+----------------+    final_output is already a str)
               |
               v
+-----------------------------+
|  plain Python, no LLM       |
|  build_chart_payload()      |   unchanged
|  markdown_to_safe_html()    |
+--------------+----------------+
               |
               v
   JSON response to the
   frontend (unchanged shape)
```

## Glossary (for a beginner reading the other files)

- **Agent** — a reusable *description* of an LLM call: a name, the rules it
  should follow (`instructions`), which model to use, and optionally what
  shape its answer must be (`output_type`). Creating an `Agent` object does
  **not** call OpenAI. It's like defining a class — just configuration.
- **Runner.run_sync(agent, input)** — the thing that actually calls OpenAI.
  You give it an `Agent` and some input text, and it sends the request,
  waits for the response, and returns a `RunResult`.
- **`RunResult.final_output`** — the answer. If the `Agent` has no
  `output_type`, this is a plain `str`. If the `Agent` has `output_type=SomePydanticModel`,
  this is already a validated instance of that model — no manual JSON
  parsing needed.
- **`output_type`** — tells the model "your answer must match this exact
  shape," using a Pydantic model (a Python class listing typed fields).
  OpenAI enforces this at the API level (Structured Outputs), so unlike the
  current `json.loads()` + markdown-fence-stripping approach, it cannot come
  back malformed.
- **`instructions`** — the fixed, unchanging part of what you tell the
  model (its "role" and rules). This maps to the parts of the current
  prompt strings that don't depend on the specific question being asked.
- **`input`** (the second argument to `Runner.run_sync`) — the part that
  changes every call: the user's actual question, or (for the answer agent)
  the question plus the fetched data.
