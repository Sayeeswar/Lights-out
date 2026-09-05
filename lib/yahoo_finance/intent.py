"""
Ask the LLM which Yahoo Finance capability a user's question requires.
"""

import json

if __package__:
    from .config import MODEL, YAHOO_MODULES, client
else:
    from config import MODEL, YAHOO_MODULES, client


def detect_intent(question: str) -> dict:
    """
    Ask the LLM which Yahoo Finance capability is required,
    and return it as a parsed dict.
    """

    prompt = f"""

You are a stock-market intent router.
Determine which Yahoo Finance data is required to answer
the user's question.
If they are two companies add then to the ticker list.
Check if two or  more modules are to be compared if yes then return the module and submodule of both the modules to be compared.
For modules like cashflow and income_stmt, return the specific row label(s) being asked about (e.g. "Free Cash Flow", "Net Income", "Total Revenue", "Operating Expenses") and store them in the submodule list.
If multiple row labels are being compared within cashflow, balance_sheet, or income_stmt, return all of them in the submodule list.
For "history", do NOT put dates or row labels in submodule — leave submodule as an empty list, and instead express the time range using "parameters" with yfinance's own period/interval values, e.g. {{"period": "6mo", "interval": "1d"}}. Valid period values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Valid interval values: 1d, 1wk, 1mo.
For modules with no row-level structure at all (e.g. "info", "news", "dividends", "recommendations"), leave submodule as an empty list.

User question:
{question}
Available Yahoo Finance modules:
{json.dumps(YAHOO_MODULES, indent=2)}
Return ONLY valid JSON.
Example Format:
{{
    "ticker": ["RELIANCE.NS,TCS.NS"],
    "module": ["cashflow", "history"],
    "submodule": ["Free Cash Flow", "Net Income"],
    "parameters": {{"period": "6mo", "interval": "1d"}},
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
Do not invent a submodule row label that is clearly implausible for a financial statement — use standard, commonly recognized line items.
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

