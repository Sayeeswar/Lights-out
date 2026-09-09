"""
Ask the LLM which Yahoo Finance capability a user's question requires.

One LLM call, via the OpenAI Agents SDK. The routing rules and the module
list are baked into the Agent's instructions once at import time; only the
question varies per call. output_type=Intent makes OpenAI return
schema-valid JSON (Structured Outputs) -- there is no json.loads or
markdown-fence cleanup any more.
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
    Only meaningful for "history" (time range as yfinance period/interval).
    For every other module both fields come back None and detect_intent()
    strips them.

    Can't be a plain dict: Structured Outputs strict mode has no way to
    express "an object with arbitrary keys."
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
    Ask the LLM which Yahoo Finance capability is required, and return it
    as a plain dict (never an Intent object -- pipeline.py mutates the
    result with intent["module"] = ...).
    """
    result = Runner.run_sync(_intent_agent, question)
    data = result.final_output.model_dump()

    # Structured Outputs always fills parameters as
    #   {"period": <str|None>, "interval": <str|None>}
    # -- both keys present, both None for anything but a "history"
    # question. fetch.py._normalize_parameters merges it as
    #   {**{"period": DEFAULT_PERIOD, "interval": DEFAULT_INTERVAL},
    #    **(intent.get("parameters") or {})}
    # so a {"period": None, "interval": None} here would overwrite the
    # defaults with None. charting.py and answer.py also read
    # intent["parameters"] directly. Strip the Nones so the returned shape
    # matches the pre-migration contract: {} when there's nothing real to
    # pass, {"period": ..., "interval": ...} otherwise. `is not None`
    # (not plain truthiness) mirrors fetch.py._normalize_parameters.
    data["parameters"] = {
        key: value
        for key, value in data["parameters"].items()
        if value is not None
    }

    return data
