# Thinking-process UI — Overview

## What this document set is

A precise, complete specification for making the app **show its work**: while
a question is being answered, the frontend should display the pipeline's
progress — "detecting what data you need", "fetching Yahoo Finance data",
"computing valuation metrics", "writing the answer" — instead of a single
static "Thinking…" spinner.

It is written so a beginner can follow it, and so a future session (or you)
can implement it from these files alone.

Read the files in this folder in order:

1. `00-overview.md` — this file. The big picture and the ground rules.
2. `01-progress-events.md` — the backend "progress event" data model and
   where each event is emitted in `lib/yahoo_finance/pipeline.py`.
3. `02-transport.md` — how those events travel from the Flask backend to the
   browser (the endpoint, the wire format, the hosting-platform caveats).
4. `03-frontend.md` — how `public/` renders the events as a live "thinking"
   panel, and how it falls back when streaming is unavailable.
5. `04-testing-and-rollout.md` — the order to build this in, behind a flag,
   and how to verify each step.
6. `05-non-goals-and-risks.md` — what this feature deliberately does NOT do,
   and the specific things that could go wrong.
7. `06-open-decisions.md` — three decisions intentionally left open: the
   transport mechanism, how much detail to show, and whether the thinking
   panel is saved into chat history. **This is the "let's think about it
   afterwards" file.** Everything else in the spec is written to work
   whichever way those three land.

## Relationship to the Agents SDK migration

This feature is a **follow-up to** `docs/agents-sdk-migration/`, not part of
it. Build order:

- **First**, land the `openai` → `openai-agents` migration (Option A) exactly
  as that spec describes. It does not change what the app does; it only
  changes how the two LLM calls are made internally.
- **Then** build this. The token-by-token streaming option for the "writing
  the answer" stage (see `06-open-decisions.md`) is expressed in terms of the
  Agents SDK's `Runner.run_streamed(...)` API, so it assumes the migration is
  already in place. The coarser "stage labels only" behaviour does **not**
  depend on the SDK at all and could be built first if you wanted.

If you are reading this before the migration is done: you can still implement
everything except the token-streaming variant of the `answer` stage.

## The one-sentence summary

`ask_stock_ai(question)` currently runs the whole pipeline silently and
returns one JSON blob at the end. This feature adds an **optional** progress
channel: the same pipeline, unchanged in what it computes, also emits a small
typed event at the start and end of each stage, a new streaming endpoint
relays those events to the browser, and the frontend draws them as a
checklist that fills in live.

## The ground rules (from `.claude/CLAUDE.md`)

Every change in this spec follows from these:

1. **The backend owns all logic.** The list of stages, their human-readable
   labels, their order, and what counts as "done" are decided in Python and
   sent to the browser as data. The frontend never hard-codes a stage name
   or infers progress on its own.
2. **The frontend is visual-only and decoupled.** It renders whatever events
   arrive. It calls no external API, holds no prompt text, contains no
   business logic. A frontend change for looks must not change behaviour.
3. **`POST /api/ask` does not change.** Its request shape, its response shape
   (`question`, `intent`, `answer`, `answer_html`, `charts`), and its status
   codes stay byte-for-byte as they are today. The progress feature is
   additive: a **new** endpoint, and a **new optional argument** to
   `ask_stock_ai`. With the feature turned off, the app behaves exactly as it
   does now.
4. **Nothing sensitive goes on the wire.** No API keys, no full prompt or
   `instructions` text, no raw model reasoning, no dumped Yahoo Finance
   payloads. Progress `detail` carries only summaries — a ticker, a list of
   module names, a count of fields. See `01-progress-events.md`.
5. **`public/` and `lib/` are write-off-limits for Claude.** This spec lives
   in `docs/`. The files it changes —
   `lib/yahoo_finance/pipeline.py` (and the two agent modules if you take the
   streaming option), `public/js/*`, `public/style.css` — must be edited by
   **you**, or by a session explicitly told the restriction is lifted for
   this task. `api/ask.py` is **not** inside `lib/` or `public/`, so the new
   endpoint there is an edit Claude can make directly.

## The core concept: a "progress event"

A progress event is a tiny dict describing one thing the pipeline is doing:

```json
{
  "stage":  "fetch",
  "status": "done",
  "label":  "Fetched Yahoo Finance data",
  "detail": { "ticker": ["AAPL"], "modules": ["info"], "field_counts": { "info": 142 } },
  "seq":    3,
  "ts":     1725800000.12
}
```

- `stage` — a fixed machine name (`intent`, `fetch`, `valuation`, `answer`, …).
- `status` — `running`, `done`, `skipped`, or `error`.
- `label` — the human sentence to show. **Written in Python**, not the browser.
- `detail` — a small, safe summary object (see rule 4). May be `{}`.
- `seq` — a monotonic counter so the client can order/dedupe.
- `ts` — a Unix timestamp, for a future "took 1.4s" annotation.

`01-progress-events.md` defines the full stage list and the exact `detail`
shape per stage.

## The diagram

```
  User question
       |
       v
+-------------------------------------------------------------+
|  ask_stock_ai(question, on_progress=None)                    |   lib/yahoo_finance/pipeline.py
|                                                             |
|  on_progress is a callback. When None (today's default,     |
|  and what POST /api/ask still passes), the pipeline runs    |
|  silently and behaves EXACTLY as it does now.               |
|                                                             |
|  detect_intent ......... emit intent   running -> done      |
|  valuation force-add ... emit intent_adjust (or skipped)    |
|  execute_yahoo_intent .. emit fetch    running -> done      |
|  reorder_statement_data  emit reshape  running -> done/skip |
|  compute_valuation_metrics emit valuation running -> done/skip|
|  generate_answer ....... emit answer   running -> done      |
|  build_chart_payload ... emit charts   running -> done      |
+----------------------------+--------------------------------+
                             |  events via on_progress(event_dict)
                             v
+-------------------------------------------------------------+
|  POST /api/ask/stream   (new; api/ask.py)                    |
|  relays each event to the browser over the wire.            |
|  The FINAL message carries the exact same dict that         |
|  POST /api/ask returns today.                                |
+----------------------------+--------------------------------+
                             |  (transport: see 06-open-decisions.md)
                             v
+-------------------------------------------------------------+
|  public/js/thinking.js (new) + ask.js + render.js            |
|  draws a checklist that fills in live, then (optionally)     |
|  collapses to a "Thinking (N steps)" summary.                |
|                                                             |
|  If the stream fails for ANY reason, ask.js falls back to    |
|  the existing POST /api/ask path — no visible error.         |
+-------------------------------------------------------------+
```

## Glossary

- **Progress event** — the small dict above. The unit this whole feature
  moves around.
- **Stage** — one step of the pipeline that gets its own event(s). There are
  seven; `01-progress-events.md` lists them.
- **`on_progress`** — a new optional parameter on `ask_stock_ai`. A function
  the pipeline calls with each event. Default `None` = emit nothing = today's
  behaviour.
- **Streaming endpoint** — the new `POST /api/ask/stream` (name provisional;
  see `06-open-decisions.md`). Returns many messages over one request instead
  of one JSON body.
- **SSE (Server-Sent Events)** — a simple standard for a server pushing a
  sequence of text events down one long-lived HTTP response
  (`Content-Type: text/event-stream`). The recommended transport, but not yet
  decided — see `06-open-decisions.md`.
- **Thinking panel** — the new frontend UI element that lists the stages.
- **Fallback** — the frontend behaviour when streaming is unavailable: use
  the old `POST /api/ask` and show the current single spinner.
