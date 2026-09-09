# Backend: the progress-event model

**Files this describes:** `lib/yahoo_finance/pipeline.py` (the emit points),
and — only if you take the token-streaming option in `06-open-decisions.md` —
`lib/yahoo_finance/answer.py`.
**Depended on by:** `api/ask.py` (relays events) and `public/js/thinking.js`
(renders them).

Both `pipeline.py` and `answer.py` are inside `lib/`, so Claude can write
this spec but not apply it — see rule 5 in `00-overview.md`.

## The design in one paragraph

`ask_stock_ai` gains one optional parameter, `on_progress`. It is a function
the pipeline calls with a plain event dict at the boundaries of each stage.
When `on_progress` is `None` — the default, and what `POST /api/ask` keeps
passing — **not a single line of behaviour changes**: no events are built, no
timestamps are read, the function returns the same dict it returns today. The
progress feature is something you only pay for when you ask for it.

## The new signature

```python
# lib/yahoo_finance/pipeline.py

from typing import Callable, Optional

ProgressFn = Callable[[dict], None]

def ask_stock_ai(question: str, on_progress: Optional[ProgressFn] = None) -> dict:
    ...
```

- `ask_stock_ai("...")` — unchanged. Returns the same dict as today.
- `ask_stock_ai("...", on_progress=fn)` — same return value, and `fn(event)`
  is called ~13 times along the way (running + done for most stages).

`on_progress` must never be allowed to break the pipeline. Wrap each call:

```python
def _emit(on_progress, stage, status, label, detail=None, _seq=[0]):
    if on_progress is None:
        return
    _seq[0] += 1
    try:
        on_progress({
            "stage": stage,
            "status": status,
            "label": label,
            "detail": detail or {},
            "seq": _seq[0],
            "ts": time.time(),
        })
    except Exception:
        # A broken progress sink must not take the answer down with it.
        pass
```

> **Learn-by-doing opportunity (later).** The `_seq` counter above uses the
> mutable-default-argument trick to keep state across calls without a class.
> That works but is widely considered a footgun. When you implement this, a
> good 5-line exercise is to replace `_emit` with a small `Progress` helper
> object (or a closure) created once per `ask_stock_ai` call, so the sequence
> counter is per-request rather than per-process. Decide: what happens to
> `seq` across two concurrent requests with the current version?

## The seven stages

Mapped against the current `pipeline.py` (`ask_stock_ai`, lines ~25–58):

| # | `stage` | Pipeline code today | `running` label | `done` label | Can be `skipped`? |
|---|---|---|---|---|---|
| 1 | `intent` | `intent = detect_intent(question)` | "Working out what data your question needs" | "Identified the data to fetch" | no |
| 2 | `intent_adjust` | the `_wants_valuation(...)` block that may `modules.append("info")` | "Checking for valuation data" | "Added company info for valuation" | **yes** — emit `skipped` when it's not a valuation question |
| 3 | `fetch` | `yahoo_data = execute_yahoo_intent(intent)` | "Fetching Yahoo Finance data" | "Fetched Yahoo Finance data" | no |
| 4 | `reshape` | the `for module, data in yahoo_data.items(): … reorder_statement_data(...)` loop | "Ordering the financial statements" | "Ordered the financial statements" | **yes** — `skipped` when no statement modules are present |
| 5 | `valuation` | the `compute_valuation_metrics(info)` block | "Computing valuation metrics" | "Computed valuation metrics" | **yes** — `skipped` when `info` is absent/empty |
| 6 | `answer` | `answer = generate_answer(question, intent, yahoo_data)` | "Writing the answer" | "Answer ready" | no |
| 7 | `charts` | `charts = build_chart_payload(intent, yahoo_data)` | "Preparing charts" | "Charts ready" / "No charts for this question" | no (but `done` label varies on `len(charts)`) |

Emit order per stage: one `running` event when the stage begins, then one
`done` (or `skipped`) event when it finishes. `answer` may additionally emit
`answer_delta` events in between — see the last section.

## The `detail` payload per stage — an allowlist

`detail` is the only place data from inside the pipeline reaches the browser.
Treat it as an **allowlist**: these keys and nothing else. When in doubt,
send a count, not the thing being counted.

| `stage` | `detail` on `done` | Explicitly NOT included |
|---|---|---|
| `intent` | `{"ticker": intent["ticker"], "module": intent["module"], "submodule": intent["submodule"], "reason": intent["reason"]}` | the prompt, the `instructions`, the raw model text |
| `intent_adjust` | `{"added": ["info"]}` or `{}` on skip | — |
| `fetch` | `{"ticker": [...], "modules": [...], "field_counts": {mod: len(data) or n_rows}}` | the fetched values themselves — only sizes |
| `reshape` | `{"modules": [names reordered]}` | statement contents |
| `valuation` | `{"metrics": [name for name in metrics]}` (the ratio names only) | the computed numbers, the `info` dict |
| `answer` | `{"chars": len(answer)}` | the answer text is delivered by the final `done` message, not here |
| `charts` | `{"count": len(charts)}` | chart data points |

Redaction rule, stated once: **if a key's value could contain a company's
actual financial figures, a prompt, a key, or free-form model output, it does
not belong in `detail`.** A reviewer should be able to look at any event and
see only routing metadata.

## Where the final answer goes

The streaming transport (next doc) sends a terminal message that carries the
**exact dict `ask_stock_ai` returns today**:

```json
{ "question": "...", "intent": {...}, "answer": "...",
  "answer_html": "...", "charts": [...] }
```

So the pipeline's `return {...}` at the end is unchanged. The progress events
are strictly extra. A client that ignores every `step` event and only reads
the terminal message gets precisely what `POST /api/ask` gives it.

## Optional: token-by-token for the `answer` stage

Only relevant if `06-open-decisions.md` lands on "stream the answer text".
This is the one part that needs the Agents SDK migration in place.

Today (post-migration) `generate_answer` ends with:

```python
result = Runner.run_sync(_answer_agent, dynamic_input)
return result.final_output
```

To stream, add a **sibling** function — do not change `generate_answer`'s
`-> str` contract, `POST /api/ask` still calls it:

```python
# lib/yahoo_finance/answer.py  (new function, next to generate_answer)

async def generate_answer_streamed(question, intent, yahoo_data, on_delta):
    """Same inputs as generate_answer. Calls on_delta(text_chunk) as the
    model produces text, and returns the full string at the end."""
    dynamic_input = _build_dynamic_input(question, intent, yahoo_data)  # factor out of generate_answer
    result = Runner.run_streamed(_answer_agent, dynamic_input)
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            data = event.data
            # token deltas on the Responses API stream:
            if getattr(data, "type", None) == "response.output_text.delta":
                on_delta(data.delta)
    return result.final_output
```

Notes / caveats (mirroring the migration spec's tone about SDK churn):

- `Runner.run_streamed`, `result.stream_events()`, the `event.type` strings
  and `response.output_text.delta` are the Agents SDK / Responses API names
  at the time of writing. **Re-verify against the installed version** — if
  any name fails, `python -c "import agents, inspect; print([n for n in dir(agents.Runner)])"`
  and check the SDK changelog. Do not guess replacements.
- `run_streamed` is async; the sync pipeline calls it via `asyncio.run(...)`
  inside the stream endpoint's generator, **not** from inside the existing
  sync `ask_stock_ai`. Keep the async surface at the edge.
- If streaming raises for any reason, catch it and fall back to
  `generate_answer` (the sync path). The user still gets an answer, just not
  token-by-token.
- The `emit` for `answer` becomes: `running` → many `answer_delta`
  (`detail: {"text": chunk}`) → `done`.

## What must NOT change

- `ask_stock_ai(question)` with no second argument: same return dict, same
  side effects, same exceptions. `POST /api/ask` keeps calling it this way.
- `detect_intent`, `execute_yahoo_intent`, `reorder_statement_data`,
  `compute_valuation_metrics`, `generate_answer`, `build_chart_payload` —
  signatures and return types untouched. Progress is emitted **around** them,
  never by changing them (except adding the new sibling `generate_answer_streamed`).
- The order of stages and the shape of the final returned dict.

## How to verify

Python commands — per this repo's rules, hand them to the user to run with
the `! <command>` prefix.

```
# 1. Old callers unaffected:
! python -c "from lib.yahoo_finance import ask_stock_ai; r = ask_stock_ai('What is Apple\'s P/E ratio?'); print(sorted(r))"
#   expect: ['answer', 'answer_html', 'charts', 'intent', 'question']

# 2. Callback fires in order:
! python -c "
from lib.yahoo_finance import ask_stock_ai
seen = []
ask_stock_ai('Show Reliance cash flow for 3 years', on_progress=lambda e: seen.append((e['stage'], e['status'])))
for row in seen: print(row)
"
#   expect: intent running / intent done / intent_adjust ... / fetch running /
#           fetch done / reshape ... / valuation ... / answer running /
#           answer done / charts running / charts done

# 3. No secrets in detail — eyeball every event's 'detail':
! python -c "
from lib.yahoo_finance import ask_stock_ai
ask_stock_ai('Is Microsoft overvalued?', on_progress=lambda e: print(e['stage'], e['detail']))
"
#   expect: only tickers, module-name lists, counts. No numbers from the
#   financial data, no prompt text.
```
