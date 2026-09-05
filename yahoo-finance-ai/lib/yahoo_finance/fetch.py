"""
Execute a detected intent against the yfinance API and return
JSON-safe data.
"""

import yfinance as yf

from .json_safety import make_json_safe

DEFAULT_PERIOD = "10y"
DEFAULT_INTERVAL = "1d"


def _fetch_one_module(stock, module: str, parameters: dict):
    if module == "fast_info":
        return make_json_safe(dict(stock.fast_info))

    elif module == "info":
        return make_json_safe(stock.info)

    elif module == "history":
        period = parameters.get("period", DEFAULT_PERIOD)
        interval = parameters.get("interval", DEFAULT_INTERVAL)
        if period == "max":
            period = DEFAULT_PERIOD
        return make_json_safe(stock.history(period=period, interval=interval))

    elif module == "income_stmt":
        return make_json_safe(stock.income_stmt)

    elif module == "quarterly_income_stmt":
        return make_json_safe(stock.quarterly_income_stmt)

    elif module == "balance_sheet":
        return make_json_safe(stock.balance_sheet)

    elif module == "quarterly_balance_sheet":
        return make_json_safe(stock.quarterly_balance_sheet)

    elif module == "cashflow":
        return make_json_safe(stock.cashflow)

    elif module == "quarterly_cashflow":
        return make_json_safe(stock.quarterly_cashflow)

    elif module == "dividends":
        dividends = stock.dividends
        if dividends.empty:
            return make_json_safe(dividends)

        dividends_2023_2025 = dividends.loc[
            (dividends.index >= "2023-01-01")
            & (dividends.index < "2026-01-01")
        ]
        return make_json_safe(dividends_2023_2025)

    elif module == "splits":
        return make_json_safe(stock.splits)

    elif module == "actions":
        return make_json_safe(stock.actions)

    elif module == "recommendations":
        return make_json_safe(stock.recommendations)

    elif module == "analyst_price_targets":
        return make_json_safe(stock.analyst_price_targets)

    elif module == "quarterly_earnings":
        # NOTE: Ticker.earnings / quarterly_earnings is deprecated upstream.
        # Prefer income_stmt's "Net Income" row where available.
        try:
            return make_json_safe(stock.quarterly_earnings)
        except Exception:
            stmt = stock.quarterly_income_stmt
            if "Net Income" in stmt.index:
                return make_json_safe(stmt.loc["Net Income"])
            return make_json_safe(stmt)

    elif module == "earnings_dates":
        limit = parameters.get("limit", 12)
        return make_json_safe(stock.get_earnings_dates(limit=limit))

    elif module == "earnings_estimate":
        return make_json_safe(stock.earnings_estimate)

    elif module == "earnings_history":
        return make_json_safe(stock.earnings_history)

    elif module == "institutional_holders":
        return make_json_safe(stock.institutional_holders)

    elif module == "major_holders":
        return make_json_safe(stock.major_holders)

    elif module == "insider_transactions":
        return make_json_safe(stock.insider_transactions)

    elif module == "insider_roster_holders":
        return make_json_safe(stock.insider_roster_holders)

    elif module == "options":
        return make_json_safe(stock.options)

    elif module == "news":
        return make_json_safe(stock.news)

    elif module == "sec_filings":
        return make_json_safe(stock.sec_filings)

    elif module == "calendarEvents":
        return make_json_safe(stock.calendar)

    else:
        raise ValueError(f"Unsupported Yahoo Finance module: {module}")


def execute_yahoo_intent(intent: dict):
    # intent.py enforces exactly one company per question.
    # Works for only one company make it work for more than one company.
    ticker_symbol = intent["ticker"][0]
    modules = intent["module"]
    parameters = {
        "period": DEFAULT_PERIOD,
        "interval": DEFAULT_INTERVAL,
        **(intent.get("parameters") or {}),
    }

    if parameters["period"] == "max":
        parameters["period"] = DEFAULT_PERIOD

    stock = yf.Ticker(ticker_symbol)

    return {
        module: _fetch_one_module(stock, module, parameters)
        for module in modules
    }
