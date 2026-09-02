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


# ============================================================
# Configuration
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Reads OPENAI_API_KEY from the environment automatically.
# On Vercel: set this in Project Settings -> Environment Variables.
client = OpenAI()


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

    else:
        raise ValueError(f"Unsupported Yahoo Finance submodule: {submodule}")


# ============================================================
# Generate final answer
# ============================================================

def generate_answer(question: str, intent: dict, yahoo_data) -> str:

    data_json = json.dumps(yahoo_data, indent=2, ensure_ascii=False)

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the Yahoo Finance
data supplied below.

User question:
{question}

Detected intent:
{json.dumps(intent, indent=2)}

Yahoo Finance data:
{data_json}

Instructions:

1. Answer the actual question directly.
2. Do not claim information that is not present in the Yahoo Finance data.
3. For financial statements, explain the relevant metric rather than
   dumping the entire statement.
4. If the user asks "cash made", determine whether the appropriate
   metric is Operating Cash Flow, Free Cash Flow, Cash Flow from
   Investing, Cash Flow from Financing, or Change in Cash.
5. If the question is ambiguous, explain the interpretation you are using.
6. Preserve the units reported by Yahoo Finance when possible.
7. If the data contains several years, compare them.
8. If a value is missing or null, say that the Yahoo Finance data
   does not provide it.
9. Do not fabricate numbers.

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
    answer = generate_answer(question, intent, yahoo_data)
    chart = build_chart_payload(intent, yahoo_data)

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "chart": chart,  # None when not applicable
    }
