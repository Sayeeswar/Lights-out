# Testing and rollout plan

Build this in the order below. Each step leaves the app fully working — if
you stop after any step, nothing is broken and the feature is simply less
complete. Do **not** build all four layers and test at the end.

## Before you start

- The Agents SDK migration (`docs/agents-sdk-migration/`) should be landed
  first if you want the token-streamed answer. If it isn't, build everything
  except that one option — the coarse "stage labels" panel doesn't touch the
  LLM calls.
- Any command that runs Python (`python`, `pip`, `flask`, `gunicorn`,
  `pytest`) is handed to **you** to run with the `! <command>` prefix — Claude
  does not run them. `git`, `curl`, browser checks are fine to run normally.
- Two flags gate the whole feature:
  - backend: `ENABLE_PROGRESS_STREAM=1` (env var; unset ⇒ new route 404s)
  - frontend: `USE_STREAM = true` in `public/js/config.js`
  With either off, the app is exactly today's app.

## Step 0 — capture a baseline

Save the current `POST /api/ask` responses for three questions that exercise
the tricky stages (same three the migration spec uses):

```
! curl -s -X POST http://localhost:3000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"What is Apple's P/E ratio?\"}" > base_pe.json
! curl -s -X POST http://localhost:3000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"Show me Reliance's cash flow for the last 3 years\"}" > base_cf.json
! curl -s -X POST http://localhost:3000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"Show me TCS stock price history for 6 months\"}" > base_hist.json
```

These are the "did the answer change?" reference for Step 4.

## Step 1 — `on_progress` in the pipeline, default off

Apply the `ask_stock_ai(question, on_progress=None)` change and the `_emit`
calls from `01-progress-events.md`. Nothing else.

Verify old callers are untouched, then the callback:

```
! python -c "from lib.yahoo_finance import ask_stock_ai; print(sorted(ask_stock_ai('What is Apple\'s P/E ratio?')))"
#   -> ['answer', 'answer_html', 'charts', 'intent', 'question']   (identical to today)

! python -c "
from lib.yahoo_finance import ask_stock_ai
ev = []
ask_stock_ai('Show me Reliance\'s cash flow for the last 3 years', on_progress=lambda e: ev.append((e['seq'], e['stage'], e['status'])))
[print(x) for x in ev]
"
#   -> seq strictly increasing; stages in pipeline order;
#      every non-skippable stage has running THEN done.
```

Re-run the Step 0 curls — the JSON must be **byte-identical** to the
baselines (this path passes `on_progress=None`). If it isn't, the emit code
changed behaviour somewhere it shouldn't have.

Run the existing test suite if there is one (`! pytest`), or at least the
migration spec's verification commands.

## Step 2 — the streaming endpoint, behind the flag

Add `POST /api/ask/stream` to `api/ask.py` per `02-transport.md` (or the
alternative transport chosen in `06-open-decisions.md`). Set
`ENABLE_PROGRESS_STREAM=1` locally.

```
! curl -N -X POST http://localhost:3000/api/ask/stream -H "Content-Type: application/json" -d "{\"question\": \"Is Microsoft overvalued?\"}"
```

Check:

- `event: step` lines appear **one at a time over several seconds**, not all
  at once at the end. (If they clump: worker class first — run gunicorn with
  `--worker-class gthread --threads 8` — then `X-Accel-Buffering`.)
- Exactly one terminal `event: done`, and its `data:` is a JSON object with
  the same keys as `base_pe.json` etc.
- With `ENABLE_PROGRESS_STREAM` unset, the route returns `404` and
  `POST /api/ask` still works.
- No API key, prompt text, or financial figures anywhere in the `step`
  `detail` payloads — read them.

Then deploy to a Render preview and run the **same curl against the deployed
URL** (and against the Vercel-proxied `/api/ask/stream`). This is where proxy
buffering shows up; localhost won't tell you.

## Step 3 — the frontend panel, behind `USE_STREAM`

Add `public/js/thinking.js`, the `ask.js` restructure, `render.js`
`renderThinking`, the `storage.js` load-repair line, the `index.html` script
tag, and the `style.css` classes — all per `03-frontend.md`.

With `USE_STREAM = false` first: the app must look and behave **exactly** as
today. Diff the DOM if unsure. No panel, no console noise.

Then `USE_STREAM = true` against the flag-on backend:

- Valuation question → panel fills in, `intent_adjust` shows "Added company
  info…", `charts` shows the no-charts label, answer renders unchanged.
- History question → `reshape` and `valuation` render as **skipped**, chart
  renders.
- Stop the backend mid-answer → fallback to `/api/ask` or the
  dropped-connection error you chose; never a stuck spinner.
- DevTools → Network → the stream request shows as `eventsource`/streaming
  with a growing response, not one final chunk.

## Step 4 — compare answers to the baseline

Re-run the Step 0 curls against `POST /api/ask` (the non-stream route) and
diff against `base_*.json`. The answer text can be phrased differently only
if the Agents SDK migration is what changed it — the progress feature itself
must not move the answer at all, because `/api/ask` never touches
`on_progress`.

Also grab the stream's terminal `done` payload for the same three questions
and confirm its `intent` / `answer_html` / `charts` match what `/api/ask`
returns for that question (allowing for LLM phrasing variance).

## Step 5 — decide the open questions, then finish

Only now, with the machinery proven, resolve `06-open-decisions.md`:
transport, depth of detail, persistence. Each has a small experiment
described there. Implement the choices and re-run Steps 2–4 for anything that
changed.

## Rollback

- **Instant, no deploy:** unset `ENABLE_PROGRESS_STREAM`, or ship
  `USE_STREAM = false`. Feature gone, app normal.
- **Code:** the touched files are
  `lib/yahoo_finance/pipeline.py`, `api/ask.py`, `public/js/ask.js`,
  `public/js/render.js`, `public/js/storage.js`, `public/js/config.js`,
  `public/js/thinking.js` (new), `public/index.html`, `public/style.css`
  (plus `lib/yahoo_finance/answer.py` if you took the streamed-answer
  option). All tracked by git.

  ```
  ! git status
  ! git diff -- <the files above>
  ! git checkout -- <the files above>   # and: rm public/js/thinking.js
  ```

  Confirm with `git status` first — never a blanket `git checkout .` that
  could discard the unrelated in-progress work already in this tree.
