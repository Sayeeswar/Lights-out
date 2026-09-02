# Yahoo Finance AI Assistant — Vercel backend

An HTTP API version of the original CLI script. One endpoint:

```
POST /api/ask
Content-Type: application/json

{ "question": "How much cash did Reliance make last year?" }
```

Response:

```json
{
  "question": "...",
  "intent": { "ticker": "RELIANCE.NS", "module": "ticker", "submodule": "cashflow", "parameters": {}, "reason": "..." },
  "answer": "..."
}
```

## UI

`public/index.html` is a single static page — dark-themed chat-style box
that calls `POST /api/ask` and renders the answer, ticker, and submodule
used. No build step, no framework; open the deployed root URL
(`https://<your-project>.vercel.app/`) and it's just there.

## What changed from the CLI version

- The `input()` / `while True` console loop is gone — replaced by a Flask
  route (`api/ask.py`) that Vercel's Python runtime serves as a serverless
  function.
- Core logic moved into `lib/yahoo_finance.py` so it's a plain importable
  module with no side effects at import time (required for serverless —
  every request may cold-start a fresh process).
- `python-dotenv` / `load_dotenv()` removed — Vercel injects environment
  variables directly; `.env` files aren't used in production. Kept
  `.env.example` for local development only.
- Fixed the earlier bugs from testing: `.output_text` + `json.loads()`
  parsing in `detect_intent()`, `make_json_safe()` (not `json()`) in
  `execute_yahoo_intent()`, and swapped the deprecated `stock.earnings`
  for a safe fallback onto `income_stmt`'s "Net Income" row.
- Errors now return a JSON `{"error": "..."}` with HTTP 500 instead of
  being printed to a console.

## Deploy

1. Push this folder to a GitHub repo (or run `vercel` from inside it
   with the Vercel CLI installed).
2. In the Vercel dashboard, import the repo as a new project.
3. Under **Settings → Environment Variables**, add:
   - `OPENAI_API_KEY` — required
   - `OPENAI_MODEL` — optional, defaults to `gpt-5`
4. Deploy. Your endpoint will be live at:
   `https://<your-project>.vercel.app/api/ask`

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
python -m flask --app api/ask run --port 3000
```

Then:
```bash
curl -X POST http://localhost:3000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Apple'\''s free cash flow trend?"}'
```

## Notes / limits

- yfinance calls happen on every request (no caching layer here) — for
  production traffic, consider adding a short-lived cache (e.g. Vercel KV
  or an in-memory TTL cache) in front of `execute_yahoo_intent()`.
- Vercel serverless functions have execution time limits by plan; `maxDuration`
  is set to 30s in `vercel.json` — raise it (Pro/Enterprise) if slower
  Yahoo Finance responses cause timeouts.
- CORS isn't configured — add `flask-cors` if this API will be called
  from a browser on a different origin.
