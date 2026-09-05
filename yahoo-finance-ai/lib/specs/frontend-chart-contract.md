# Spec: public/js contract changes for multi-chart / histogram support

Status: design agreed via chat on 2026-09-05, **all three files now
implemented** the same day (`ask.js`, `render.js`, `chart.js`), matching
the backend contract from `charting-histogram-spec.md`. Verified with
`node --check` on all three (syntax only) — not yet visually confirmed in a
live browser against the running backend, since that requires the Python
backend process, which is a hand-off, not something run directly here.

Depends on `charting-histogram-spec.md` producing a **list** of chart dicts,
where statement-row charts (cashflow/income_stmt) carry a `series` list and
everything else keeps the flat `{x, y}` shape.

## `ask.js` — DONE (implemented earlier)

- `message.chart: null` → `message.charts: []` (initial message shape).
- `message.chart = data.chart || null;` → `message.charts = data.charts || [];`
  (reading the API response).

This is safe to ship ahead of the backend: until the backend actually sends
`data.charts`, `data.charts || []` just yields an empty array, so nothing
renders — no crash.

## `render.js` — DONE

Loops `m.charts` (an array), creating one canvas per entry via
`pendingCharts`, so e.g. `module: ["cashflow", "history"]` → 2 chart
objects → 2 `<canvas>` elements under that answer, in whatever order the
backend list puts them in.

Also added a `displayList()` helper so the ticker/submodule badges join
array values with `", "` instead of relying on JS's default array-to-string
coercion (which drops the space after each comma) — covers both
`intent.ticker` and `intent.submodule`, since both are lists now.

## `chart.js` — DONE

`renderChart` now delegates dataset construction to `buildDatasets(chart)`:

- If `chart.series` is present (statement-row charts): builds one Chart.js
  dataset per `series` entry, all sharing `chart.x` as labels, cycling
  through a small 4-color `CHART_PALETTE` so multiple bars per period are
  visually distinguishable. With `type: "bar"` and multiple dataset
  objects, Chart.js groups them side-by-side per x-axis category
  automatically — no extra grouping config was needed. Works correctly
  even when `series` has only one entry (renders as an ordinary
  single-series bar chart, matching Rule 2 in `charting-histogram-spec.md`).
  Line-only styling (`tension`, `pointRadius`, `fill`) is intentionally
  left off this branch since every `series` chart is `chart_type: "bar"`.
- Else (`chart.y` present — history/dividends/splits): unchanged, same
  single-dataset logic as before.

Axis/tooltip logic (`monthYear`, scales, tooltip title callback) is shape-
agnostic and did not need to change.

## Open item

Once `render.js` and `chart.js` are updated, re-verify in a real browser
that a two-module question (e.g. the "compare cash flow + show history"
example) renders two canvases correctly, and that a single-row
cashflow/income_stmt question still renders a normal (non-grouped) bar
chart via the `series`-of-length-1 path.
