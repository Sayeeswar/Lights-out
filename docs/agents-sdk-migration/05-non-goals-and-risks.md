# Non-goals and risks

## Explicit non-goals (do not bundle these into this migration)

These were discussed and deliberately excluded from Option A. If you want
any of them, treat them as separate follow-up work with their own testing
pass — don't mix them into the same change as this spec, because it makes
it much harder to tell which change caused a regression.

- **Option B (the "idiomatic single agent" design).** Turning
  `execute_yahoo_intent` into a tool the model calls itself inside a
  multi-turn loop. This trades away the deterministic guarantees
  `pipeline.py` currently enforces in plain `if` statements (e.g.
  `_wants_valuation()` always force-adding `"info"`), and replaces a fixed,
  predictable 2-call latency shape with a variable, model-decided number of
  calls. Out of scope here entirely.
- **Multi-ticker fetching.** `fetch.py` has a comment flagging that
  `execute_yahoo_intent()` only ever reads `intent["ticker"][0]`, even
  though `intent.py`'s prompt already asks the model to list multiple
  tickers for comparison questions. This spec's `Intent.ticker` stays
  `list[str]` (matching today), but nothing in this migration makes
  `fetch.py` actually loop over more than the first ticker. That's a
  `fetch.py`/`pipeline.py` data-shape change, independent of which OpenAI
  SDK is used.
- **Parallel/concurrent fetching.** Whether fetching multiple modules (or
  multiple tickers) concurrently with threads. Pure Python, unrelated to
  this migration, safe to do before or after it.

## Files this migration does NOT touch, and why

| File | Why it's untouched |
|---|---|
| `lib/yahoo_finance/fetch.py` | Takes a `dict`, returns a `dict`. Doesn't care how the `dict` it receives was produced. |
| `lib/yahoo_finance/valuation.py` | Pure computation on a `dict`. No LLM involvement at all. |
| `lib/yahoo_finance/statement_ordering.py` | Pure computation. `answer.py` imports `STATEMENT_SUBMODULES` from it as a constant — unchanged. |
| `lib/yahoo_finance/charting.py` | Pure computation on `intent` + `yahoo_data` dicts. Doesn't know or care they came from an `Agent`. |
| `lib/yahoo_finance/pipeline.py` | Calls `detect_intent()` and `generate_answer()` by their existing names and expects their existing return types — both preserved exactly. |
| `api/ask.py` | Imports `ask_stock_ai` from the package `__init__.py`. Never imports OpenAI-specific anything directly. |
| `lib/yahoo_finance.py` (the standalone file, not the package) | This is dead code — `api/ask.py` imports `lib.yahoo_finance`, which Python resolves to the *package* (`lib/yahoo_finance/__init__.py`), not this loose file. It still uses the raw `OpenAI` client and is **not** part of the live request path. Confirmed by reading `lib/yahoo_finance/__init__.py`, which only re-exports `ask_stock_ai` from `.pipeline`. Left alone — not in scope, and it's inside `lib/` regardless. |

## Known risks, ranked by how likely they are to bite

1. **The `parameters` None-stripping fix in `intent.py` gets skipped or
   miscopied.** This is the single most concrete, already-identified risk
   in this spec — see the full explanation in `02-intent-module.md`. The
   failure mode is specific and testable: a non-history question's
   `parameters` field ends up as `{"period": None, "interval": None}`
   instead of `{}`, which corrupts `fetch.py`'s default-merging logic for
   *any* question, not just history ones (because the merge happens before
   the module-specific branch is even chosen).

2. **The `openai-agents` package's API has moved since this spec was
   written.** This spec's code (`Agent`, `Runner.run_sync`,
   `set_default_openai_client`, `output_type`) was checked against the
   package's own README and docs at the time of writing, not assumed from
   memory — but this is a young, actively-developed SDK. Specifically
   worth re-checking after `pip install`:
   - Run `python -c "import agents; print(agents.__version__)"` (hand off
     to the user) and compare against the SDK's changelog if any of the
     import names in this spec fail.
   - If `set_default_openai_client` doesn't exist in the installed
     version, search the installed package for the current equivalent
     (`python -c "import agents; print([n for n in dir(agents) if 'client' in n.lower()])"`)
     rather than guessing.

3. **Strict Structured Outputs schema errors on `Intent` at runtime.**
   This spec's `Parameters` submodel exists specifically to avoid the
   "arbitrary dict" problem, but if a schema-validation error still occurs
   when you actually run it, the error message from the OpenAI API is
   usually explicit about which field/constraint failed (commonly
   mentioning `additionalProperties` or a field being outside the
   `required` list) — read that message before changing anything else in
   the model; it will point at the exact field, not require guessing.

4. **`gpt-5` (or whatever `OPENAI_MODEL` is set to) behaves slightly
   differently under Structured Outputs than it did under free-text +
   manual parsing.** Structured Outputs constrains the model more tightly,
   which occasionally changes *content* choices at the margins (e.g. how
   it fills a field when genuinely unsure), even though it can't produce
   an invalid shape anymore. This is why `04-testing-and-rollout.md` asks
   you to compare real question/answer pairs before and after, not just
   check that the code runs without exceptions.

## The permission boundary, restated

This spec exists as a set of files under `docs/agents-sdk-migration/`
specifically because the three files it describes changing
(`lib/yahoo_finance/config.py`, `intent.py`, `answer.py`) are inside
`lib/`, which this repository's `.claude/CLAUDE.md` marks off-limits for
Claude to write to. If you (or a future session) want Claude to apply
these changes directly instead of you copying them in by hand, that
requires either changing that policy for this task or explicitly
authorizing the specific edit in the moment — Claude won't infer that
permission from this spec file existing.
