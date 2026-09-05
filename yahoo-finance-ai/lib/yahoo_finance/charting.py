"""
Turn cleaned Yahoo Finance data into a small shape a frontend chart
library (e.g. Chart.js) can plot directly.
"""

# Submodules where the cleaned data is naturally chartable, and which
# column/row to plot. "history" data is a dict of
# {date_str: {"Open": ..., "Close": ..., ...}}; "dividends"/"splits" are
# dicts of {date_str: value}; financial statements are
# {row_label: {period_str: value}}.
CHARTABLE_SUBMODULES = {
    "history": {"type": "ohlc_series", "field": "Close", "chart": "line"},
    "dividends": {"type": "flat_series", "chart": "bar"},
    "splits": {"type": "flat_series", "chart": "bar"},
    "cashflow": {"type": "statement_rows", "chart": "bar"},
    "quarterly_cashflow": {"type": "statement_row", "row": "Operating Cash Flow", "chart": "bar"},
    "income_stmt": {"type": "statement_rows", "chart": "bar"},
    "quarterly_income_stmt": {"type": "statement_row", "row": "Net Income", "chart": "bar"},
}


def _ohlc_series(data, field):
    return [
        (date_str, row.get(field))
        for date_str, row in data.items()
        if isinstance(row, dict) and row.get(field) is not None
    ]


def _flat_series(data):
    return [(date_str, value) for date_str, value in data.items() if value is not None]


def _statement_row(data, row):
    row_data = data.get(row)
    if not row_data:
        return None
    return [(period_str, value) for period_str, value in row_data.items() if value is not None]


def _statement_rows_chart(data, rows, label, chart_type):
    """
    Build a {x, series} payload comparing one or more rows of the same
    financial statement (e.g. Operating Cash Flow vs Free Cash Flow).
    Works the same whether `rows` has one entry or several - the caller
    never needs to special-case the count.
    """
    row_datas = {}
    all_periods = set()
    for row in rows:
        row_data = data.get(row)
        if row_data:
            row_datas[row] = row_data
            all_periods.update(row_data.keys())

    if not row_datas:
        return None

    periods = sorted(all_periods)
    series = [
        {"label": row, "data": [row_datas[row].get(period) for period in periods]}
        for row in rows
        if row in row_datas
    ]

    return {
        "chart_type": chart_type,
        "label": label,
        "x": [str(p) for p in periods],
        "series": series,
    }


def build_chart_payload(intent: dict, yahoo_data):
    """
    Turn yahoo_data (a dict keyed by module name, as produced by
    execute_yahoo_intent) into a list of small chart shapes a frontend chart
    library (e.g. Chart.js) can plot directly. A module that isn't
    chartable, or whose expected shape/field isn't present in the data, is
    simply left out of the list. Returns an empty list when nothing in the
    intent is chartable.
    """

    if not isinstance(yahoo_data, dict) or not yahoo_data:
        return []

    ticker = ", ".join(intent.get("ticker", []) or [])
    submodule_rows = intent.get("submodule", []) or []
    charts = []

    for module in intent.get("module", []) or []:
        spec = CHARTABLE_SUBMODULES.get(module)
        data = yahoo_data.get(module)

        if not spec or not isinstance(data, dict) or not data:
            continue

        label = f"{ticker} — {module}"

        try:
            if spec["type"] == "statement_rows":
                chart = _statement_rows_chart(data, submodule_rows, label, spec["chart"])
                if chart:
                    charts.append(chart)
                continue

            if spec["type"] == "ohlc_series":
                items = _ohlc_series(data, spec["field"])
            elif spec["type"] == "flat_series":
                items = _flat_series(data)
            elif spec["type"] == "statement_row":
                items = _statement_row(data, spec["row"])
            else:
                items = None

            if not items:
                continue

            items.sort(key=lambda pair: pair[0])

            charts.append({
                "chart_type": spec["chart"],
                "label": label,
                "x": [str(k) for k, _ in items],
                "y": [v for _, v in items],
            })

        except Exception:
            # Charting is a bonus, never let it break the main answer.
            continue

    return charts
