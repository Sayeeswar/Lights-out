# Spec: charting.py — multi-row histogram for cashflow / income_stmt

Status: design agreed via chat on 2026-09-05, **implemented** the same day
in `lib/yahoo_finance/charting.py` and wired into `pipeline.py` (response
key renamed `chart` → `charts` to match the list return; `ask.js` already
expected `data.charts`, see `frontend-chart-contract.md`). Depended on
`pipeline-contract-gap.md`, which was resolved first.

## Goal

When a question compares two or more row labels within the same statement
(e.g. "compare Reliance's operating cash flow and free cash flow"), render
them as one grouped bar chart (a histogram: one group of bars per period,
one bar per row) instead of only ever charting a single hardcoded row.

## Scope (explicitly agreed)

- **In scope now:** `cashflow` and `income_stmt` (annual only).
- **Out of scope for now:** `quarterly_cashflow`, `quarterly_income_stmt`
  (same code path, deliberately deferred — revisit later if wanted).
- **Untouched:** `balance_sheet` (was never in `CHARTABLE_SUBMODULES` to
  begin with — no regression from leaving it alone), `history`, `dividends`,
  `splits` (all keep today's flat `{x, y}` shape; not worth the churn right
  now since none of them have a "compare multiple rows" concept).

## `CHARTABLE_SUBMODULES` change

Give `cashflow` and `income_stmt` a new type, `"statement_rows"` (plural),
replacing `"statement_row"` + a single hardcoded `"row"` key:

```python
CHARTABLE_SUBMODULES = {
    "history": {"type": "ohlc_series", "field": "Close", "chart": "line"},
    "dividends": {"type": "flat_series", "chart": "bar"},
    "splits": {"type": "flat_series", "chart": "bar"},
    "cashflow": {"type": "statement_rows", "chart": "bar"},
    "quarterly_cashflow": {"type": "statement_row", "row": "Operating Cash Flow", "chart": "bar"},
    "income_stmt": {"type": "statement_rows", "chart": "bar"},
    "quarterly_income_stmt": {"type": "statement_row", "row": "Net Income", "chart": "bar"},
}
```

`quarterly_*` entries keep their old single-row behavior unchanged (out of
scope, see above).

## Data shape rules

### Rule 1 — `build_chart_payload` returns a list, not a single dict/`None`

Today it returns one dict or `None` for a single assumed module. Once
`intent["module"]` can hold two entries (e.g. `["cashflow", "history"]`),
the function must build one chart entry per chartable module present and
return them as a list (empty list when nothing is chartable). Example, for
`module: ["cashflow", "history"]`, `submodule: ["Operating Cash Flow", "Free Cash Flow"]`:

```python
[
    {  # from the cashflow module
        "chart_type": "bar",
        "label": "RELIANCE.NS — cashflow",
        "x": ["2021-03-31", "2022-03-31", "2023-03-31", "2024-03-31"],
        "series": [
            {"label": "Operating Cash Flow", "data": [120000, 135000, 128000, 150000]},
            {"label": "Free Cash Flow", "data": [80000, 95000, 60000, 110000]},
        ],
    },
    {  # from the history module — untouched flat shape
        "chart_type": "line",
        "label": "RELIANCE.NS — history",
        "x": ["2025-03-01", "2025-03-02", "..."],
        "y": [2891.4, 2905.0, "..."],
    },
]
```

### Rule 2 — `"statement_rows"` always uses `series`, even for one row

No branch on "1 row vs many rows" — a single-row cashflow/income_stmt
question still produces `series` with exactly one entry:

```python
{
    "chart_type": "bar",
    "label": "RELIANCE.NS — cashflow",
    "x": ["2021-03-31", "2022-03-31", "2023-03-31", "2024-03-31"],
    "series": [
        {"label": "Free Cash Flow", "data": [80000, 95000, 60000, 110000]},
    ],
}
```

This keeps the consumer (`chart.js`, per `frontend-chart-contract.md`) from
needing a count-based special case — it always reads `series`.

### Rule 3 — everything else keeps the old flat shape

`"ohlc_series"` (history), `"flat_series"` (dividends/splits), and the
`"statement_row"` (singular) path used by the deferred quarterly entries all
keep returning `{chart_type, label, x, y}` exactly as today.

## Resolved implementation details

- `yahoo_data` is the dict keyed by module name that `execute_yahoo_intent`
  now produces (per `pipeline-contract-gap.md`). `build_chart_payload` keeps
  its `(intent, yahoo_data)` signature unchanged; it just looks up
  `yahoo_data.get(module)` per module in `intent["module"]`.
- Row alignment: resolved as **union** of periods across the requested
  rows, sorted ascending, shared as one `x` list. A row missing a given
  period gets `None` in its `series` entry's `data` at that position (via
  `row_datas[row].get(period)`), rather than that row's points shifting out
  of alignment with the others. A row missing from the data entirely is
  dropped from `series` (not padded with an all-`None` series).
- `build_chart_payload` returns a **list**, built by looping
  `intent["module"]`; a module that isn't chartable, or whose data is
  missing/empty, is skipped (`continue`) rather than aborting the whole
  list — matches the existing "charting is a bonus" philosophy, just
  applied per-module instead of per-call.
- Side fix noticed while implementing: the `quarterly_cashflow` /
  `quarterly_income_stmt` entries in the previous `CHARTABLE_SUBMODULES`
  had `"row": "{Based_on_context}"` — a literal placeholder string that
  could never match a real row label, so those two were already always
  producing `None` (no chart) before today. Replaced with concrete
  defaults (`"Operating Cash Flow"`, `"Net Income"`) matching the older
  flat `lib/yahoo_finance.py`'s values, so they now actually chart
  something, consistent with keeping them out of the multi-row scope for
  now rather than leaving them silently broken.
