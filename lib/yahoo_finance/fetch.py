"""
Fetch the Yahoo Finance data a detected intent asks for (single ticker).

The pipeline downstream (pipeline.py, charting.py) works on a flat
{module: data} dict for ONE company. The intent router may still name
several tickers for a comparison question; _select_ticker() decides which
one this single-ticker pipeline fetches.

A question can still name several *modules* ("cashflow and history").
yfinance calls are blocking network I/O, so the modules are fetched
concurrently on a thread pool (asyncio.to_thread + gather), bounded by
MAX_CONCURRENT_REQUESTS. execute_yahoo_intent() is the sync entry point
pipeline.py imports; it wraps the async runner in asyncio.run().
"""

import asyncio
from typing import Any

import yfinance as yf

from .json_safety import make_json_safe

DEFAULT_PERIOD = "10y"
DEFAULT_INTERVAL = "1d"
MAX_CONCURRENT_REQUESTS = 5


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


def _select_ticker(intent: dict) -> str:
    """
    Reduce intent["ticker"] (always a list; the router may list more than
    one for a comparison question, and older prompts sometimes emit a
    single comma-joined string like "RELIANCE.NS,TCS.NS") down to the ONE
    ticker symbol this single-ticker pipeline will fetch.

    Comparison questions keep working: we always take the FIRST company and
    answer for it, rather than raising.
    """
    tickers = intent.get("ticker") or []
    if not tickers:
        raise ValueError("Intent has no ticker to fetch.")
    first = str(tickers[0]).split(",")[0].strip()
    if not first:
        raise ValueError("Intent's first ticker is empty.")
    return first


def _normalize_parameters(intent: dict) -> dict:
    """history / earnings_dates parameters, with defaults applied.

    None values are stripped first: a structured-output intent can carry
    {"period": None, "interval": None} for a non-history question, and a
    plain ** spread of that would clobber the defaults with None (then
    stock.history(period=None) misbehaves).
    """
    supplied = {
        key: value
        for key, value in (intent.get("parameters") or {}).items()
        if value is not None
    }
    parameters = {
        "period": DEFAULT_PERIOD,
        "interval": DEFAULT_INTERVAL,
        **supplied,
    }
    if parameters["period"] == "max":
        parameters["period"] = DEFAULT_PERIOD
    return parameters


async def _fetch_module_async(
    stock, module: str, parameters: dict, semaphore: asyncio.Semaphore,
) -> tuple[str, Any]:
    """Run the blocking _fetch_one_module() on a worker thread.

    yfinance is synchronous network I/O (requests under the hood) and does
    not cooperate with the event loop. asyncio.to_thread() hands the call
    to a thread pool so other modules' fetches run while this one waits on
    the network. "async def" alone would not make it concurrent -- the
    thread offload is what does.
    """
    async with semaphore:
        try:
            data = await asyncio.to_thread(
                _fetch_one_module, stock, module, parameters
            )
        except Exception as exc:
            data = {"error": str(exc)}
    return module, data


async def _execute_yahoo_intent_async(intent: dict) -> dict[str, Any]:
    """Concurrent multi-module fetch for the one selected ticker.

    Returns {module: data}. A module whose fetch raised comes back as
    {"error": "..."} instead of failing the whole run. A task that blew up
    before it could report which module it was is collected under a
    top-level "_errors" list.
    """
    ticker_symbol = _select_ticker(intent)
    modules = list(intent["module"])
    parameters = _normalize_parameters(intent)

    stock = yf.Ticker(ticker_symbol)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [
        asyncio.create_task(
            _fetch_module_async(stock, module, parameters, semaphore)
        )
        for module in modules
    ]

    # return_exceptions=True: one failed module must not cancel the rest.
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: dict[str, Any] = {}
    for outcome in task_results:
        if isinstance(outcome, Exception):
            results.setdefault("_errors", []).append(str(outcome))
            continue
        module, data = outcome
        results[module] = data
    return results


def execute_yahoo_intent(intent: dict) -> dict[str, Any]:
    """Sync entry point for the pipeline.

    Runs the concurrent per-module fetch for a single ticker and returns a
    flat {module: data} dict -- the shape pipeline.py and charting.py
    expect. Safe to call from a plain sync request handler: asyncio.run()
    builds and tears down its own event loop, and the Agents SDK's
    Runner.run_sync (intent / answer) runs at other points in the pipeline
    with its own loop, never nested inside this one.
    """
    return asyncio.run(_execute_yahoo_intent_async(intent))


