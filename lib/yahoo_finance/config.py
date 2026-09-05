"""
Shared configuration: the OpenAI client/model and the map of Yahoo Finance
capabilities the intent router is allowed to choose from.
"""
# imprt dotenv package in python
import os

from dotenv import load_dotenv

from openai import OpenAI
load_dotenv()  # Load environment variables from .env file
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Reads OPENAI_API_KEY from the environment automatically.
# Set it in the Render dashboard -> Environment.
# Extra retries/timeout so a transient network blip on the host doesn't
# surface as APIConnectionError on the first (cold-start) request.
client = OpenAI(timeout=60.0, max_retries=4)


YAHOO_MODULES = {
    "ticker": [
        "fast_info",
        "history",
        "info",

        # Financial statements
        "income_stmt" ,
        "quarterly_income_stmt",
        "balance_sheet",
        "quarterly_balance_sheet",
        "cashflow" ,
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
        "calendar",
        "sec_filings",
    ]
}
