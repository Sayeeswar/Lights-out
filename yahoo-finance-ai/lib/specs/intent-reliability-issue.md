# Issue: LLM intent detection is not reliable enough for the histogram feature

Status: discovered during manual testing on 2026-09-05, after
`charting-histogram-spec.md` and `frontend-chart-contract.md` were both
implemented and confirmed working at least once. Not yet fixed. No code
changed as part of writing this document.

## What was observed

Testing "Compare Reliance's operating cash flow and free cash flow"
repeatedly against the local backend:

- Sometimes it works exactly as designed: histogram renders, submodule
  tags show `Operating Cash Flow` and `Free Cash Flow`.
- Sometimes the histogram doesn't render, and the submodule tag shows the
  literal word **"cashflow"** instead of a real row label.

The second case is a strong signal, not just "no chart appeared." The
submodule badges (`render.js`) display exactly whatever strings are in
`intent.submodule` — nothing invented on the frontend. A tag reading
"cashflow" means `detect_intent()` (the LLM call in `intent.py`) itself
returned something like `submodule: ["cashflow"]` for that call — i.e. it
put a **module name** where a **statement row label** belongs.

## Why this breaks the histogram specifically

`_statement_rows_chart` (in `charting.py`) does `data.get(row)` for each
entry in `submodule`, where `data` is `yahoo_data["cashflow"]` — a dict
shaped `{row_label: {period: value}}`. Its keys are real statement rows
like `"Operating Cash Flow"`, `"Free Cash Flow"`, `"Cash Flow from
Investing"`, etc. There is no row literally called `"cashflow"`. So:

- The lookup finds nothing for `"cashflow"`.
- `row_datas` ends up empty.
- `_statement_rows_chart` returns `None` (correctly, per its own contract
  — this is the code doing exactly what it's supposed to when handed a row
  label that doesn't exist).
- No chart for that module gets appended to the `charts` list.

So the charting code is not the bug here. The root cause is one level up:
`detect_intent()`'s output isn't consistent from call to call for what
should be the same classification.

## Root cause

`intent.py`'s prompt asks an LLM to produce `module` (the Yahoo Finance
capability, e.g. `"cashflow"`) and `submodule` (specific row labels within
that capability, e.g. `"Operating Cash Flow"`) as separate concepts. LLM
sampling isn't perfectly consistent question to question — occasionally it
collapses these two concepts and answers with the module name in both
places, or in `submodule` alone. This is a classification-reliability
problem inherent to using an LLM for this step, not a deterministic parsing
bug.

## Not yet confirmed

The specific `intent.module` / `intent.submodule` values for a *failing*
request haven't been captured yet (via the browser's Network tab). The
"cashflow" tag is strong circumstantial evidence, but the next session
should confirm the exact bad `intent` shape before picking a fix, in case
there's a second, different failure mode hiding behind the same symptom.

## Candidate fixes (not yet chosen, not yet implemented)

These are independent — either or both could be done:

1. **Tighten `intent.py`'s prompt** so the LLM is less likely to conflate
   "which module" with "which row inside that module." Could mean adding
   an explicit negative example (e.g. "submodule must never equal a module
   name like 'cashflow' or 'income_stmt'"), or restructuring the prompt so
   the two fields are asked for more distinctly.
2. **Add defensive handling in `charting.py`** for a `submodule` entry that
   doesn't match any real row in the data — options range from silently
   skipping (current behavior, safe but produces no chart) to a
   loose/fuzzy match against the real row labels, to surfacing the mismatch
   somehow instead of just omitting the chart.

## Open decision for next session

Which of the two candidate fixes to pursue first (or both), and if fixing
`intent.py`'s prompt, what the tightened wording should actually say.
