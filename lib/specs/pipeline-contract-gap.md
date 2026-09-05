# Spec: intent.py contract vs. fetch.py/pipeline.py (blocking gap)

Status: discovered during charting-histogram design discussion, 2026-09-05.
**Fixed** the same day in `fetch.py` and `pipeline.py` (see "Resolution"
below). `charting.py` still has the old single-string assumption — that's
tracked separately in `charting-histogram-spec.md`.

## Current `intent.py` output shape

```
{
  "ticker": ["RELIANCE.NS"],           // list, even for one company
  "module": ["cashflow", "history"],   // list of Yahoo Finance capabilities
  "submodule": ["Operating Cash Flow", "Free Cash Flow"],
                                        // list of row labels, ONLY meaningful
                                        // for statement modules (cashflow,
                                        // income_stmt, balance_sheet, and
                                        // their quarterly variants); empty
                                        // list for everything else (history,
                                        // dividends, info, news, ...)
  "parameters": {"period": "6mo", "interval": "1d"},
  "reason": "..."
}
```

Key point: `module` names the Yahoo Finance capability. `submodule` names
*rows within a statement*, not a capability. A module with no row-level
structure (history, dividends, info, ...) always has `submodule: []`.

## Where the rest of the pipeline still assumes the old shape

Both of these currently treat `intent["submodule"]` as **a single string
equal to a module name** (e.g. `"cashflow"`), which was the *old* contract
before `intent.py` was rewritten to the list-based shape above.

### `lib/yahoo_finance/fetch.py` — `execute_yahoo_intent(intent)`

```python
submodule = intent["submodule"]
...
elif submodule == "cashflow":
    return make_json_safe(stock.cashflow)
```

Under the new contract, `intent["submodule"]` is a list of row labels (or an
empty list) — it is never equal to `"cashflow"`. Every `elif` branch fails
and the function falls through to:

```python
else:
    raise ValueError(f"Unsupported Yahoo Finance submodule: {submodule}")
```

So today, calling `execute_yahoo_intent` with output from the current
`intent.py` raises `ValueError` for every question — nothing can be fetched.

### `lib/yahoo_finance/pipeline.py` — `ask_stock_ai(question)`

```python
submodule = intent.get("submodule")
if submodule in STATEMENT_SUBMODULES:
    annual = not submodule.startswith("quarterly_")
    yahoo_data = reorder_statement_data(yahoo_data, annual=annual)
```

Same issue: `submodule` is a list, so `submodule in STATEMENT_SUBMODULES`
(a set of module-name strings) is always `False`, and `submodule.startswith(...)`
would raise `AttributeError` on a list if that branch were ever entered.

## Likely cause

Git status shows `lib/yahoo_finance/` (the package) as untracked/new,
alongside the still-tracked, still-modified `lib/yahoo_finance.py` (flat
legacy module). Read: a refactor from one flat file into the package is in
progress. `intent.py` was updated to the new list-based contract first;
`fetch.py` and `pipeline.py` (and `statement_ordering.py`'s usage inside
`pipeline.py`) have not caught up yet. The old flat `lib/yahoo_finance.py`
still has the *original* single-row `CHARTABLE_SUBMODULES`/`build_chart_payload`
and the same old `submodule`-as-module-name assumption — consistent with
this being the pre-refactor version kept as reference.

## What needs to change here (not yet done)

This is a prerequisite for `charting-histogram-spec.md` to work end-to-end,
not something the charting spec can work around:

1. **`fetch.py`**: `execute_yahoo_intent` needs to iterate `intent["module"]`
   (now a list) and fetch each named capability, returning data keyed by
   module name (e.g. `{"cashflow": {...}, "history": {...}}`) rather than
   one flat blob for a single assumed module.
2. **`pipeline.py`**: `ask_stock_ai` needs to check statement-ness per module
   (iterate `intent["module"]`, check each name against `STATEMENT_SUBMODULES`)
   rather than checking the whole `submodule` list as if it were one module
   name. `reorder_statement_data` would need to run per statement module in
   the `module` list, not once against a single `yahoo_data` blob.

## Resolution

Both fixed on 2026-09-05:

- **`fetch.py`**: the old `submodule == "cashflow"` chain was extracted into
  `_fetch_one_module(stock, module, parameters)`, unchanged branch-for-branch
  except the variable is now named `module` (it always was a module name, not
  a "submodule"). `execute_yahoo_intent` now does
  `ticker_symbol = intent["ticker"][0]` (intent.py enforces exactly one
  company) and loops `intent["module"]`, returning
  `{module_name: data, ...}` — a dict keyed by module name, uniformly, even
  when only one module was requested. This confirms the shape
  `charting-histogram-spec.md` assumed.
- **`pipeline.py`**: replaced the single `submodule in STATEMENT_SUBMODULES`
  check with a loop over `yahoo_data.items()`, applying
  `reorder_statement_data` per module that is itself a statement type,
  in place.

`generate_answer` (`answer.py`) needed no change — it only `json.dumps`s
whatever `yahoo_data` it receives, with no shape assumptions of its own, so
it works unchanged against the new per-module-keyed dict.

`charting.py` is the one remaining file still assuming the old
single-string `submodule` contract — see `charting-histogram-spec.md` for
that work, not yet done.
