# Module spec: `intent.py`

**File this describes:** `lib/yahoo_finance/intent.py`
**Depended on by:** `pipeline.py`, which calls `detect_intent(question)` and
then does `intent.get("module")`, `intent["module"] = modules`,
`intent.get("reason", "")` on the result.

This is the most detailed file in this spec, because this is the module
where a real compatibility trap exists (see "A gotcha you need to know
about," below). Read this one fully before applying it.

## What this file does today

One function, `detect_intent(question: str) -> dict`:

1. Builds a large prompt string by hand, embedding the question and the
   full `YAHOO_MODULES` list (serialized to JSON) every single call.
2. Calls `client.responses.create(model=MODEL, input=prompt)`.
3. Takes `response.output_text`, strips it, and calls `json.loads()` on it.
4. If that raises `json.JSONDecodeError` (e.g. the model wrapped the JSON in
   ` ```json ` fences), strips the fences and tries `json.loads()` again.

The shape the prompt asks for (and that downstream code depends on):

```json
{
    "ticker": ["RELIANCE.NS", "TCS.NS"],
    "module": ["cashflow", "history"],
    "submodule": ["Free Cash Flow", "Net Income"],
    "parameters": {"period": "6mo", "interval": "1d"},
    "reason": "Why this Yahoo Finance capability is required"
}
```

## What changes, and why

Instead of describing this shape in prose inside the prompt and hoping the
model's text output matches it, you describe it as a **Pydantic model** —
an actual Python class with typed fields — and pass it as the `Agent`'s
`output_type`. The Agents SDK turns that into a JSON Schema and asks OpenAI
to constrain its output to match it exactly (this is OpenAI's "Structured
Outputs" feature). The result: `json.loads()` and the markdown-fence
fallback aren't needed anymore, because the API guarantees valid,
schema-matching JSON — there's no "hope it parses" step.

The static parts of today's prompt (the routing rules, numbered 1–20, and
the embedded `YAHOO_MODULES` JSON) become the `Agent`'s `instructions` —
built **once**, when the module is imported, not re-built inside the
function on every call. The only thing that varies per call is the
question itself, which becomes the `input` to `Runner.run_sync(...)`.

## A gotcha you need to know about (read this before applying the change)

OpenAI's Structured Outputs feature, in its strict mode (which is what
`output_type` uses), does **not** support a field whose type is "a dict
with arbitrary keys" — like today's `"parameters": {}`, which can hold
`period`/`interval` for `history` questions or be empty for everything
else. Strict mode requires every field's shape to be fully and exactly
declared in advance; it can't say "this object can have any keys."

So `parameters` can't just be typed as a plain `dict` in the Pydantic
model. It needs to be its own small model with named, optional fields:

```python
class Parameters(BaseModel):
    period: str | None = None
    interval: str | None = None
```

That solves the schema problem, but creates a second, subtler one. Look at
how `fetch.py` (which this migration does **not** touch, and can't touch)
actually reads `parameters`:

```python
# lib/yahoo_finance/fetch.py — execute_yahoo_intent()
parameters = {
    "period": DEFAULT_PERIOD,
    "interval": DEFAULT_INTERVAL,
    **(intent.get("parameters") or {}),
}
```

This relies on **the key being absent** for a non-`history` question, so
the `DEFAULT_PERIOD`/`DEFAULT_INTERVAL` values win. But with the
`Parameters` model above, a non-`history` question won't produce a missing
key — it'll produce `{"period": None, "interval": None}`, a dict that
**is** truthy (so `or {}` does not kick in) and **does** have both keys.
The `**` spread would then overwrite `DEFAULT_PERIOD`/`DEFAULT_INTERVAL`
with `None`, and later, `stock.history(period=None, ...)` would very likely
break or silently misbehave.

**The fix:** after getting the structured result back, strip out any
`None` values from `parameters` before returning the dict, so the returned
shape matches exactly what a non-`history` question produces today (an
empty dict, no keys at all). This is a small, local fix entirely inside
`detect_intent()` — it does not require touching `fetch.py`. It's included
in the "full new file content" below.

## Full new file content

```python
"""
Ask the LLM which Yahoo Finance capability a user's question requires.
"""

import json

from pydantic import BaseModel
from agents import Agent, Runner

if __package__:
    from .config import MODEL, YAHOO_MODULES
else:
    from config import MODEL, YAHOO_MODULES


class Parameters(BaseModel):
    """
    Only meaningful for "history" questions. Left as None/None (stripped to
    an empty dict before detect_intent() returns) for every other module —
    see the note in this file's docstring-equivalent spec doc about why
    this can't just be a plain dict under Structured Outputs strict mode.
    """
    period: str | None = None
    interval: str | None = None


class Intent(BaseModel):
    ticker: list[str]
    module: list[str]
    submodule: list[str] = []
    parameters: Parameters = Parameters()
    reason: str


_ROUTER_INSTRUCTIONS = f"""
You are a stock-market intent router.
Determine which Yahoo Finance data is required to answer
the user's question.
If they are two companies add then to the ticker list.
Check if two or  more modules are to be compared if yes then return the module and submodule of both the modules to be compared.
For modules like cashflow and income_stmt, return the specific row label(s) being asked about (e.g. "Free Cash Flow", "Net Income", "Total Revenue", "Operating Expenses") and store them in the submodule list.
If multiple row labels are being compared within cashflow, balance_sheet, or income_stmt, return all of them in the submodule list.
For "history", do NOT put dates or row labels in submodule — leave submodule as an empty list, and instead express the time range using "parameters" with yfinance's own period/interval values, e.g. {{"period": "6mo", "interval": "1d"}}. Valid period values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Valid interval values: 1d, 1wk, 1mo.
For modules with no row-level structure at all (e.g. "info", "news", "dividends", "recommendations"), leave submodule as an empty list.

Available Yahoo Finance modules:
{json.dumps(YAHOO_MODULES, indent=2)}

Rules:
1. For Indian stocks, use the NSE ticker when appropriate.
   Example:
   Reliance Industries -> RELIANCE.NS
   TCS -> TCS.NS
   Infosys -> INFY.NS
2. For US stocks:
   Apple -> AAPL
   Microsoft -> MSFT
3. Use "history" for historical prices; express the date range via "parameters" (period/interval), never in submodule.
4. Use "info" for company information.
5. Use "income_stmt" for annual income statements.
6. Use "quarterly_income_stmt" for quarterly income statements.
7. Use "balance_sheet" for annual balance sheets.
8. Use "quarterly_balance_sheet" for quarterly balance sheets.
9. Use "cashflow" for annual cash flow.
10. Use "quarterly_cashflow" for quarterly cash flow.
11. Use "dividends" for dividend history.
12. Use "recommendations" for analyst recommendations.
13. Use "analyst_price_targets" for analyst price targets.
14. Use "quarterly_earnings" for quarterly earnings.
15. Use "options" for options chains.
16. Use "news" for company news.
17. If the user asks something ambiguous such as
    "cash that Reliance made", interpret it as cash flow
    and use cashflow.
18. Use "calendarEvents" for upcoming calendar events (next earnings
    date, dividend / ex-dividend dates).
19. Use "sec_filings" for SEC filings.
20. Use "info" for any valuation question — whether a stock is overvalued,
    undervalued, or fairly priced, and any request for P/E, forward P/E,
    P/S, P/B, EV/EBITDA, or PEG. "info" carries the current price, market
    cap, EPS, book value, and those ratios. Always include "info" in the
    module list for such a question; also add "balance_sheet" if the user
    asks about debt in the same question.
Do not invent a submodule row label that is clearly implausible for a financial statement — use standard, commonly recognized line items.
"""

_intent_agent = Agent(
    name="Yahoo Finance Intent Router",
    instructions=_ROUTER_INSTRUCTIONS,
    model=MODEL,
    output_type=Intent,
)


def detect_intent(question: str) -> dict:
    """
    Ask the LLM which Yahoo Finance capability is required,
    and return it as a parsed dict.
    """
    result = Runner.run_sync(_intent_agent, question)
    data = result.final_output.model_dump()

    # Structured Outputs can't express "this key may be absent," so
    # Parameters always comes back with both fields present (None for
    # anything but a "history" question). Strip the Nones so the returned
    # shape matches what fetch.py's `**(intent.get("parameters") or {})`
    # merge expects: either a real {"period": ..., "interval": ...} or {}.
    data["parameters"] = {
        key: value for key, value in data["parameters"].items()
        if value is not None
    }

    return data
```

Notice what's **gone** compared to today's file: the `try/except
json.JSONDecodeError` block, the `.replace("```json", "")` fence-stripping,
and the f-string interpolation of `YAHOO_MODULES` happening on every call
(it's now baked into `_ROUTER_INSTRUCTIONS` once, at import time).

## Side-by-side: the return contract

| | Before | After |
|---|---|---|
| Return type | `dict` (from `json.loads`) | `dict` (from `.model_dump()`, then filtered) |
| `ticker` | list of strings | list of strings — unchanged |
| `module` | list of strings | list of strings — unchanged |
| `submodule` | list of strings | list of strings — unchanged |
| `parameters` | `{}` or `{"period": ..., "interval": ...}` | identical — `{}` or `{"period": ..., "interval": ...}` after the None-filter |
| `reason` | string | string — unchanged |
| Malformed JSON handling | manual retry with fence-stripping | not needed — API guarantees valid schema |

Every consumer of `detect_intent()`'s return value (`pipeline.py`,
`fetch.py`, `answer.py`, `charting.py`) sees the exact same shape as
before. This is the part of the migration where getting the shape wrong
would be easy to do accidentally and hard to notice — which is why this
file has the longest explanation in this spec.

## What must NOT change

- The wording of the routing rules (numbered 1–20). They encode real
  product decisions (e.g. NSE ticker suffixes, when to force `"info"` for
  valuation questions) — copy them verbatim, don't paraphrase.
- The fact that `detect_intent()` returns a plain `dict`, never an `Intent`
  object directly.
- The `parameters` None-stripping step — skipping it is the single most
  likely way this migration silently breaks `"history"`-adjacent behavior.

## How to verify

See `04-testing-and-rollout.md` for the exact command, but the core check
is: call `detect_intent("What is Apple's P/E ratio?")` and confirm the
returned dict has `"parameters": {}` (not `{"period": None, "interval":
None}`), and call it with a history-style question and confirm `parameters`
actually contains `period`/`interval` strings.
