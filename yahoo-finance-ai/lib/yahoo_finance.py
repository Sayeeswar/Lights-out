"""
Core intent-detection + Yahoo Finance fetch + answer-generation logic.

Kept deliberately free of any HTTP/CLI concerns so it can be imported
by a serverless function (api/ask.py) without change.
"""

import os
import json
import math
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

from lib.formatting import markdown_to_safe_html


# ============================================================
# Configuration
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Reads OPENAI_API_KEY from the environment automatically.
# Set it in the Render dashboard -> Environment.
# Extra retries/timeout so a transient network blip on the host doesn't
# surface as APIConnectionError on the first (cold-start) request.
client = OpenAI(timeout=60.0, max_retries=4)


# ============================================================
# Yahoo Finance capabilities
# ============================================================

YAHOO_MODULES = {
    "ticker": [
        "fast_info",
        "history",
        "info",

        # Financial statements
        "income_stmt",
        "quarterly_income_stmt",
        "balance_sheet",
        "quarterly_balance_sheet",
        "cashflow",
        "quarterly_cashflow",

        # Corporate actions
        "dividends",
        "splits",
        "actions",

        # Analyst data
        "recommendations",
        "analyst_price_targets",

        # Earnings
        "quarterly_earnings",
        "earnings_dates",
        "earnings_estimate",
        "earnings_history",

        # Ownership
        "institutional_holders",
        "major_holders",
        "insider_transactions",
        "insider_roster_holders",

        # Options
        "options",

        # News / filings
        "news",
        "calendarEvents",
        "sec_filings",
    ]
}


# ============================================================
# JSON / pandas cleaning
# ============================================================

def clean_for_json(obj):
    """
    Convert yfinance / pandas / NumPy objects into values
    that can safely be passed through json.dumps().
    """

    if obj is None:
        return None

    if isinstance(obj, (str, int, bool)):
        return obj

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.index = [clean_for_json(index) for index in df.index]
        df.columns = [clean_for_json(column) for column in df.columns]

        result = {}
        for index, row in df.iterrows():
            index_key = str(index)
            result[index_key] = {
                str(column): clean_for_json(value)
                for column, value in row.items()
            }
        return result

    if isinstance(obj, pd.Series):
        result = {}
        for index, value in obj.items():
            index_key = str(clean_for_json(index))
            result[index_key] = clean_for_json(value)
        return result

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            key = str(clean_for_json(key))
            result[key] = clean_for_json(value)
        return result

    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(value) for value in obj]

    return str(obj)


def make_json_safe(data):
    """
    Final validation layer. Ensures the returned object can
    actually be serialized with json.dumps().
    """
    cleaned = clean_for_json(data)
    json.dumps(cleaned, ensure_ascii=False)  # raises early if anything slipped through
    return cleaned


# ============================================================
# Intent detection
# ============================================================

def detect_intent(question: str) -> dict:
    """
    Ask the LLM which Yahoo Finance capability is required,
    and return it as a parsed dict.
    """

    prompt = f"""
You are a stock-market intent router.

Determine which Yahoo Finance data is required to answer
the user's question.

User question:
{question}

Available Yahoo Finance modules:

{json.dumps(YAHOO_MODULES, indent=2)}

Return ONLY valid JSON.

Format:

{{
    "ticker": "RELIANCE.NS",
    "module": "ticker",
    "submodule": "cashflow",
    "parameters": {{}},
    "reason": "Why this Yahoo Finance capability is required"
}}

Rules:

1. For Indian stocks, use the NSE ticker when appropriate.
   Example:
   Reliance Industries -> RELIANCE.NS
   TCS -> TCS.NS
   Infosys -> INFY.NS

2. For US stocks:
   Apple -> AAPL
   Microsoft -> MSFT

3. Use "history" for historical prices.
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

Do not invent a submodule that is not in the available list.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    text = response.output_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Handle accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)


# ============================================================
# Yahoo Finance execution
# ============================================================

def execute_yahoo_intent(intent: dict):

    ticker_symbol = intent["ticker"]
    submodule = intent["submodule"]
    parameters = intent.get("parameters", {})

    stock = yf.Ticker(ticker_symbol)

    if submodule == "fast_info":
        return make_json_safe(dict(stock.fast_info))

    elif submodule == "info":
        return make_json_safe(stock.info)

    elif submodule == "history":
        period = parameters.get("period", "1y")
        interval = parameters.get("interval", "1d")
        return make_json_safe(stock.history(period=period, interval=interval))

    elif submodule == "income_stmt":
        return make_json_safe(stock.income_stmt)

    elif submodule == "quarterly_income_stmt":
        return make_json_safe(stock.quarterly_income_stmt)

    elif submodule == "balance_sheet":
        return make_json_safe(stock.balance_sheet)

    elif submodule == "quarterly_balance_sheet":
        return make_json_safe(stock.quarterly_balance_sheet)

    elif submodule == "cashflow":
        return make_json_safe(stock.cashflow)

    elif submodule == "quarterly_cashflow":
        return make_json_safe(stock.quarterly_cashflow)

    elif submodule == "dividends":
        return make_json_safe(stock.dividends)

    elif submodule == "splits":
        return make_json_safe(stock.splits)

    elif submodule == "actions":
        return make_json_safe(stock.actions)

    elif submodule == "recommendations":
        return make_json_safe(stock.recommendations)

    elif submodule == "analyst_price_targets":
        return make_json_safe(stock.analyst_price_targets)

    elif submodule == "quarterly_earnings":
        # NOTE: Ticker.earnings / quarterly_earnings is deprecated upstream.
        # Prefer income_stmt's "Net Income" row where available.
        try:
            return make_json_safe(stock.quarterly_earnings)
        except Exception:
            stmt = stock.quarterly_income_stmt
            if "Net Income" in stmt.index:
                return make_json_safe(stmt.loc["Net Income"])
            return make_json_safe(stmt)

    elif submodule == "earnings_dates":
        limit = parameters.get("limit", 12)
        return make_json_safe(stock.get_earnings_dates(limit=limit))

    elif submodule == "earnings_estimate":
        return make_json_safe(stock.earnings_estimate)

    elif submodule == "earnings_history":
        return make_json_safe(stock.earnings_history)

    elif submodule == "institutional_holders":
        return make_json_safe(stock.institutional_holders)

    elif submodule == "major_holders":
        return make_json_safe(stock.major_holders)

    elif submodule == "insider_transactions":
        return make_json_safe(stock.insider_transactions)

    elif submodule == "insider_roster_holders":
        return make_json_safe(stock.insider_roster_holders)

    elif submodule == "options":
        return make_json_safe(stock.options)

    elif submodule == "news":
        return make_json_safe(stock.news)

    elif submodule == "sec_filings":
        return make_json_safe(stock.sec_filings)

    elif submodule == "calendarEvents":
        return make_json_safe(stock.calendar)

    else:
        raise ValueError(f"Unsupported Yahoo Finance submodule: {submodule}")


# ============================================================
# Statement period ordering
# ============================================================

# Submodules whose cleaned shape is {row_label: {period_str: value}} and
# therefore need their periods ordered before we hand them to the model.
STATEMENT_SUBMODULES = {
    "income_stmt", "quarterly_income_stmt",
    "balance_sheet", "quarterly_balance_sheet",
    "cashflow", "quarterly_cashflow",
}


def order_statement_periods(period_labels, today=None, annual=True, keep=6):
    """
    Given the period labels from a financial statement (ISO date strings
    such as "2025-03-31", occasionally a non-date like "TTM"), return them
    ordered so the most recent relevant period comes first.

    generate_answer() uses this so a question about "last year" resolves to
    a predictable column instead of whichever one yfinance happened to list
    first.

    Policy:
    - Labels that don't parse as a date (e.g. "TTM") are kept, but placed
      after every real dated period so they never win "latest".
    - Periods whose end date is after `today` are dropped: yfinance
      sometimes carries a column before results are actually reported.
    - For annual statements, a fiscal year that ends inside the *current*
      calendar year is treated as "this year's" report, not "last year's",
      so it is excluded from the resolved list - unless doing so would
      leave nothing, in which case the filter is skipped.
    - At most `keep` dated periods are returned, newest first.
    """
    if today is None:
        today = date.today()

    reported = []   # (end_date, label) for periods already reported
    undated = []    # labels we couldn't parse, e.g. "TTM"

    for label in period_labels:
        try:
            end = datetime.strptime(str(label)[:10], "%Y-%m-%d").date()
        except ValueError:
            undated.append(label)
            continue
        if end <= today:
            reported.append((end, label))

    before_this_year = [pair for pair in reported if pair[0].year < today.year]
    chosen = before_this_year if (annual and before_this_year) else reported

    chosen.sort(key=lambda pair: pair[0], reverse=True)
    return [label for _, label in chosen[:keep]] + undated


def reorder_statement_data(yahoo_data, today=None, annual=True):
    """
    Apply order_statement_periods() to every row of a cleaned statement so
    each row's periods share one consistent, newest-first order.
    Non-dict payloads are returned untouched.
    """
    if not isinstance(yahoo_data, dict) or not yahoo_data:
        return yahoo_data

    all_periods = set()
    for row in yahoo_data.values():
        if isinstance(row, dict):
            all_periods.update(row.keys())

    ordered = order_statement_periods(list(all_periods), today, annual=annual)

    result = {}
    for row_label, row in yahoo_data.items():
        if isinstance(row, dict):
            result[row_label] = {p: row[p] for p in ordered if p in row}
        else:
            result[row_label] = row
    return result


# ============================================================
# Generate final answer
# ============================================================

# Kept out of the f-string prompt below: the LaTeX example contains braces
# ( \frac{a}{b}, V_{1} ) that Python's f-string parser would read as fields.
ANSWER_FORMATTING_RULES = """\
Formatting (reply in GitHub-flavored Markdown, clean and skimmable):

- Use "- " bullet points for any list of figures, drivers, or comparisons -
  one point per line. Do not use a bullet for a single item.
- Add a "## " or "### " heading only when the answer has two or more
  distinct sections. Skip headings for a short answer.
- Use a Markdown table when comparing the same metric across several periods.
- Put key numbers in **bold** (e.g. **$4.2 billion**). Never bold a whole
  sentence.
- Write every mathematical expression in LaTeX: \\( ... \\) inline and
  \\[ ... \\] for a displayed equation. For example a growth rate is
  \\( \\frac{V_{1} - V_{0}}{V_{0}} \\times 100\\% \\).
- Do NOT use $ ... $ or $$ ... $$ for math: a bare $ means US dollars here.
  Write currency as plain text, e.g. $5.2 billion.
- Do not add filler such as "I hope this helps" and do not restate the
  question.
"""


def generate_answer(question: str, intent: dict, yahoo_data) -> str:

    data_json = json.dumps(yahoo_data, indent=2, ensure_ascii=False)

    prompt = f"""
You are a financial research assistant.

Today's date is {date.today().isoformat()}.

Answer the user's question using ONLY the Yahoo Finance
data supplied below.

User question:
{question}

Detected intent:
{json.dumps(intent, indent=2)}

Yahoo Finance data:
{data_json}

Instructions:.
1. Answer the actual question directly.
2. Do not claim information that is not present in the Yahoo Finance data.
3. {STATEMENT_SUBMODULES} are financial statements. Always display them in a table comparing the same metric across several periods.

4.  Always comapre Operating Cash Flow, Free Cash Flow, Cash Flow ,
   Investing, Cash Flow from Financing, and  Change in Cash in a tabular format.

5. If the data contains several years, compare them.
6. If a value is missing or null, say that the Yahoo Finance data
   does not provide it.
8. Include as much information as poosible fron yahoo finance data.
7. Do not fabricate numbers.
9. If the user says last year do {date.today().year - 1} and if the user says this year do {date.today().year}.Follow the same for yearly data.

{ANSWER_FORMATTING_RULES}
Give a concise but useful financial answer.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text


# ============================================================
# Chart data shaping
# ============================================================

# Submodules where the cleaned data is naturally chartable, and which
# column/row to plot. "history" data is a dict of
# {date_str: {"Open": ..., "Close": ..., ...}}; "dividends"/"splits" are
# dicts of {date_str: value}; financial statements are
# {row_label: {period_str: value}}.
CHARTABLE_SUBMODULES = {
    "history": {"type": "ohlc_series", "field": "Close", "chart": "line"},
    "dividends": {"type": "flat_series", "chart": "bar"},
    "splits": {"type": "flat_series", "chart": "bar"},
    "cashflow": {"type": "statement_row", "row": "Operating Cash Flow", "chart": "bar"},
    "quarterly_cashflow": {"type": "statement_row", "row": "Operating Cash Flow", "chart": "bar"},
    "income_stmt": {"type": "statement_row", "row": "Net Income", "chart": "bar"},
    "quarterly_income_stmt": {"type": "statement_row", "row": "Net Income", "chart": "bar"},
}


def build_chart_payload(intent: dict, yahoo_data):
    """
    Turn the cleaned yahoo_data into a small {label, points, chart_type}
    shape a frontend chart library (e.g. Chart.js) can plot directly.
    Returns None when the submodule isn't chartable or the expected
    shape/field isn't present in the data.
    """

    submodule = intent.get("submodule")
    spec = CHARTABLE_SUBMODULES.get(submodule)

    if not spec or not isinstance(yahoo_data, dict) or not yahoo_data:
        return None

    try:
        if spec["type"] == "ohlc_series":
            field = spec["field"]
            # yahoo_data: {date_str: {"Open":.., "Close":.., ...}}
            items = [
                (date_str, row.get(field))
                for date_str, row in yahoo_data.items()
                if isinstance(row, dict) and row.get(field) is not None
            ]

        elif spec["type"] == "flat_series":
            # yahoo_data: {date_str: value}
            items = [
                (date_str, value)
                for date_str, value in yahoo_data.items()
                if value is not None
            ]

        elif spec["type"] == "statement_row":
            # yahoo_data: {row_label: {period_str: value}}
            row_data = yahoo_data.get(spec["row"])
            if not row_data:
                return None
            items = [
                (period_str, value)
                for period_str, value in row_data.items()
                if value is not None
            ]

        else:
            return None

        if not items:
            return None

        # Sort chronologically (ISO-formatted date strings sort correctly)
        items.sort(key=lambda pair: pair[0])

        return {
            "chart_type": spec["chart"],
            "label": f"{intent.get('ticker', '')} — {submodule}",
            "x": [str(k) for k, _ in items],
            "y": [v for _, v in items],
        }

    except Exception:
        # Charting is a bonus, never let it break the main answer
        return None


# ============================================================
# Complete pipeline
# ============================================================

def ask_stock_ai(question: str) -> dict:
    """
    Runs the full pipeline and returns a dict suitable for
    a JSON HTTP response (rather than printing to a console).
    """

    intent = detect_intent(question)
    yahoo_data = execute_yahoo_intent(intent)

    submodule = intent.get("submodule")
    if submodule in STATEMENT_SUBMODULES:
        annual = not submodule.startswith("quarterly_")
        yahoo_data = reorder_statement_data(yahoo_data, annual=annual)

    answer = generate_answer(question, intent, yahoo_data)
    chart = build_chart_payload(intent, yahoo_data)

    return {
        "question": question,
        "intent": intent,
        "answer": answer,  # raw Markdown, kept for API compatibility
        "answer_html": markdown_to_safe_html(answer),  # sanitized, UI-ready
        "chart": chart,  # None when not applicable
    }
