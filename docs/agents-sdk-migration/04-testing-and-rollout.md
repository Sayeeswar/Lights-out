# Testing and rollout plan

This file is the order of operations: what to change first, what to check
after each step, and how to back out if something goes wrong. Follow it in
order — don't apply all three module changes at once and test at the end,
because if something breaks, you want to know which of the three files
caused it.

## Before you start

This app's rules (this repo's `.claude/CLAUDE.md`) require any command that
runs Python to be handed to a human to execute — Claude Code does not run
`python`, `pip`, `pytest`, `flask`, etc. itself. Every command below is
written for **you** to run, using the `! <command>` prefix in the prompt if
you're asking Claude to wait on the result.

## Step 0 — capture a baseline

Before changing anything, run a few real questions through the *current*
app and save the raw JSON responses somewhere (a scratch file, not
committed). You'll compare against these later. Example:

```
! curl -s -X POST http://localhost:3000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"What is Apple's P/E ratio?\"}" > baseline_pe.json
! curl -s -X POST http://localhost:3000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"Show me Reliance's cash flow for the last 3 years\"}" > baseline_cashflow.json
! curl -s -X POST http://localhost:3000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"Show me TCS stock price history for 6 months\"}" > baseline_history.json
```

(This assumes `flask --app api/ask run --port 3000` is already running
locally — start it first if not.)

Pick questions that exercise all three of the tricky spots this spec calls
out: a valuation question (exercises `_wants_valuation` and
`valuation_metrics`), a statement question (exercises
`STATEMENT_SUBMODULES` and table formatting), and a history question
(exercises the `parameters` field specifically — this is the one the
None-stripping fix in `02-intent-module.md` protects).

## Step 1 — add the dependency

```
! pip install openai-agents
```

Then add it to `requirements.txt` (this file is not inside `lib/` or
`public/`, so Claude can make this edit directly — ask for it, or do it
yourself):

```
openai-agents>=0.1.0
```

Pin whatever version `pip show openai-agents` reports after install, rather
than leaving it unbounded — the SDK is young and its API has already been
observed to shift between versions during research for this spec.

## Step 2 — apply `config.py`, verify in isolation

Apply the full new content from `01-config-module.md`. Then:

```
! python -c "from lib.yahoo_finance import config; print('MODEL =', config.MODEL); print('client not exported:', not hasattr(config, 'client'))"
```

Expect: no import errors, `MODEL` prints your configured model name,
`client not exported: True`. If this fails with an error about
`set_default_openai_client` not existing, the installed `openai-agents`
version doesn't match what this spec assumes — check the "How to
re-verify the SDK API" note in `05-non-goals-and-risks.md` before going
further.

## Step 3 — apply `intent.py`, verify against the baseline

Apply the full new content from `02-intent-module.md`. Then, with the
Flask app restarted:

```
! python -c "
from lib.yahoo_finance.intent import detect_intent
import json
print(json.dumps(detect_intent(\"What is Apple's P/E ratio?\"), indent=2))
print(json.dumps(detect_intent(\"Show me TCS stock price history for 6 months\"), indent=2))
"
```

Check specifically:

- The P/E question's `parameters` field is `{}` — **not**
  `{"period": null, "interval": null}`. This is the exact failure mode the
  None-stripping step exists to prevent. If you see nulls here, the strip
  step in `detect_intent()` was dropped or is buggy.
- The history question's `parameters` field actually contains real
  `period`/`interval` strings.
- `ticker`, `module`, `submodule`, `reason` are still lists/strings in the
  same shape as your Step 0 baseline's `"intent"` field.

Only move to Step 4 once this looks right — `answer.py` and
`pipeline.py` both consume whatever `detect_intent()` returns, so a
malformed `parameters` shape here will cause confusing failures downstream
that look like `answer.py`'s fault when they aren't.

## Step 4 — apply `answer.py`, compare full responses

Apply the full new content from `03-answer-module.md`. Restart the app,
then re-run the exact Step 0 curl commands against the same three
questions. Compare the new responses to `baseline_pe.json`,
`baseline_cashflow.json`, `baseline_history.json`:

- `"intent"` — same shape as Step 3 confirmed.
- `"answer"` / `"answer_html"` — not byte-identical (the model can phrase
  things differently), but check: still has a Markdown table for the
  cashflow question, still uses `\\(...\\)` LaTeX (not `$...$`) for math,
  still bolds key numbers, still well over 200 words.
- `"charts"` — still populated the same way (this comes entirely from
  `build_chart_payload`, which never changed, but it depends on `intent`
  and `yahoo_data` having the right shape, so it's a good downstream
  sanity check that nothing upstream silently broke).

## Step 5 — a quick manual pass in the actual browser UI

Load `public/index.html` against your local backend and ask it the same
three questions through the real chat UI, not just curl. This catches
anything a raw JSON diff wouldn't — e.g. the frontend choking on a field
it didn't expect.

## If something breaks: rollback

All three changed files are tracked by git. If a step fails and you want
to back out cleanly:

```
! git status
! git diff lib/yahoo_finance/config.py lib/yahoo_finance/intent.py lib/yahoo_finance/answer.py
! git checkout -- lib/yahoo_finance/config.py lib/yahoo_finance/intent.py lib/yahoo_finance/answer.py
```

Only do this after confirming with `git status`/`git diff` that these are
the only files you're discarding — don't run a broad `git checkout .`
that could discard unrelated in-progress work.
