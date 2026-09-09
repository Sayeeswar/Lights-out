# Module spec: `config.py`

**File this describes:** `lib/yahoo_finance/config.py`
**Depended on by:** `intent.py`, `answer.py` (both import `MODEL` from here;
today they also import `client`, which goes away in this migration)

## What this file does today

Three jobs, all at import time (this file runs once, when it's first
imported, and its results are reused for every request):

1. Loads `.env` so `OPENAI_API_KEY` and `OPENAI_MODEL` are available as
   environment variables.
2. Reads `MODEL` from the environment (defaults to `"gpt-4o-mini"`).
3. Builds one shared `OpenAI` client object, with a longer timeout and more
   retries than the library default — this exists specifically so a cold
   start on the hosting platform (Render) doesn't surface as a connection
   error on the very first request.
4. Defines `YAHOO_MODULES`, the list of Yahoo Finance capabilities the
   intent router is allowed to choose from. (Unrelated to OpenAI — this
   doesn't change at all.)

Current content:

```python
"""
Shared configuration: the OpenAI client/model and the map of Yahoo Finance
capabilities the intent router is allowed to choose from.
"""
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
    "ticker": [ ... ]  # full list, unchanged — see the real file
}
```

## What changes, and why

The Agents SDK's `Runner` doesn't take a client object as an argument the
way you'd call `client.responses.create(...)` directly. Instead, you
**register** a client once, globally, using `set_default_openai_client()`,
and every `Runner.run_sync(...)` call anywhere in the app automatically uses
it. This is why the change lives in `config.py`: it's the one place that
already runs exactly once, at import time, before any request comes in —
exactly the right place for one-time global setup.

Two details worth being precise about:

1. **`OpenAI` becomes `AsyncOpenAI`.** The Agents SDK is built on an async
   client internally (`Runner.run_sync` is a synchronous wrapper around an
   async call, not a truly synchronous implementation). You must pass an
   `AsyncOpenAI` instance to `set_default_openai_client`, not the `OpenAI`
   class you're using today. The constructor arguments (`timeout`,
   `max_retries`) are identical between the two classes — only the class
   name changes.
2. **There is no `client` variable to export anymore.** `intent.py` and
   `answer.py` currently do `from .config import MODEL, YAHOO_MODULES,
   client`. After this change, they'll do `from .config import MODEL,
   YAHOO_MODULES` — `client` simply won't exist as a name in this module,
   because nothing needs to hold a reference to it directly anymore; the
   Agents SDK holds it internally after registration.

## Full new file content

```python
"""
Shared configuration: registers the OpenAI client the Agents SDK will use,
and holds the model name and the map of Yahoo Finance capabilities the
intent router is allowed to choose from.
"""
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import set_default_openai_client

load_dotenv()  # Load environment variables from .env file
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# Reads OPENAI_API_KEY from the environment automatically.
# Set it in the Render dashboard -> Environment.
# Extra retries/timeout so a transient network blip on the host doesn't
# surface as APIConnectionError on the first (cold-start) request.
# Registered once here; every Runner.run_sync(...) call anywhere in the
# app (intent.py, answer.py) uses this client automatically.
set_default_openai_client(AsyncOpenAI(timeout=60.0, max_retries=4))


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
        "calendar",
        "sec_filings",
    ]
}
```

(`YAHOO_MODULES` is copy-pasted verbatim from the current file — it does not
change. Copy it exactly as it exists in the real `config.py` when you apply
this, in case it has drifted since this spec was written.)

## What must NOT change

- `MODEL`'s default value (`"gpt-40-mini"`) and the environment variable name
  (`OPENAI_MODEL`) it reads from.
- `YAHOO_MODULES`'s contents.
- The `timeout=60.0, max_retries=4` values — these were chosen deliberately
  for cold-start resilience on Render; don't drop them while switching
  client classes.

## How to verify this one change in isolation

You can't fully test this file by itself (nothing calls `Runner` yet until
`intent.py`/`answer.py` are also converted). The real check happens after
`02-intent-module.md` is applied — see `04-testing-and-rollout.md` for the
exact verification step. For now, just confirm the file **imports without
error**:

```
python -c "from lib.yahoo_finance import config"
```

This is a Python command — per this repo's rules, hand it to the user to
run with the `!` prefix rather than running it yourself.
