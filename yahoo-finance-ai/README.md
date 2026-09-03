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
  route (`api/ask.py`) served by gunicorn on Render.
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

The Python API (`pandas` + `numpy` + `yfinance`) is too heavy for a Vercel
serverless function's 250 MB limit, so it runs on **Render**; **Vercel** hosts
the static frontend and proxies `/api/*` through to Render (see
`public/vercel.json`), so the browser only ever talks to the Vercel origin.

Vercel's Root Directory is `yahoo-finance-ai/public` — pointing it there (rather
than at `yahoo-finance-ai`) keeps `api/` out of Vercel's sight so it doesn't try
to build the Flask app as a function. `.vercelignore` does not do this: it is
ignored for Git-connected deployments.

### 1. API on Render

1. Push the repo to GitHub.
2. Render → **New + → Blueprint**, point it at this repo. `render.yaml` defines
   the service (root dir `yahoo-finance-ai`, gunicorn start command,
   `/healthz` health check, Python 3.12).
3. When prompted, set `OPENAI_API_KEY` (it's marked `sync: false` so it is not
   read from the repo). `OPENAI_MODEL` and `CORS_ALLOW_ORIGIN` have defaults.
4. Note the service URL, e.g. `https://yahoo-finance-ai-api.onrender.com`.
   Free tier spins down after ~15 min idle — first request then takes 30–60s.

### 2. Frontend on Vercel

1. Edit `public/vercel.json` → replace the `destination` host with your real
   Render URL.
2. Vercel dashboard → import the repo. Set **Root Directory** to
   `yahoo-finance-ai/public`. Framework Preset: **Other**. No build command, no
   env vars (the key lives on Render).
3. Deploy. The site is at `https://<your-project>.vercel.app/`, and its
   `POST /api/ask` calls are proxied to Render.

### Alternative: skip the proxy

Instead of the `vercel.json` rewrite you can point the frontend straight at
Render — set `fetch()` in `public/index.html` to the full Render URL. CORS is
already handled (`flask-cors`, open by default; restrict with
`CORS_ALLOW_ORIGIN`).

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
