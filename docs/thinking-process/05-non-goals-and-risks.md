# Non-goals and risks

## Explicit non-goals

Do not bundle these into this feature. Each is its own change with its own
testing pass.

- **Exposing the model's actual reasoning / chain-of-thought.** The "thinking
  process" this feature shows is the *pipeline's* progress — which stage is
  running, what data was routed and fetched. It is not the LLM's internal
  reasoning tokens. Even if a future SDK version exposes reasoning traces,
  surfacing them is a separate product and privacy decision.
- **Option B (agent-as-tool) traces.** If the pipeline is ever restructured
  so one agent calls Yahoo fetching as a tool in a loop (explicitly out of
  scope in the migration spec), that produces a different, richer event
  stream (tool-call started, tool returned, model turn N…). This spec's event
  model is built for the current fixed 7-stage pipeline. Extending it to a
  dynamic agent loop is future work.
- **Server-side persistence of runs.** No database of past questions, no
  "run history" API. Progress is ephemeral on the server; the browser keeps
  whatever `06-open-decisions.md` decides to keep, in `localStorage`, as
  today.
- **Auth / rate limiting / per-user anything.** The API stays public,
  unauthenticated, read-only. Streaming doesn't change that.
- **Progress for anything other than `ask_stock_ai`.** `GET /`, `/healthz`,
  the usage stub — untouched.
- **A chat-UI redesign.** The panel slots into the existing layout and
  styling. Reworking the thread, composer, or history view is not part of
  this.
- **Cancelling a running pipeline when the browser disconnects.** Nice to
  have, meaningfully more code (thread cancellation, cooperative checks in
  `fetch.py`). Out of scope; a disconnected stream just runs to completion
  server-side and is discarded.

## Files this feature does NOT touch

| File | Why |
|---|---|
| `lib/yahoo_finance/intent.py` | Called unchanged. Events are emitted *around* `detect_intent`, not inside it. |
| `lib/yahoo_finance/fetch.py` | Called unchanged. `fetch` stage timing is measured by the pipeline, not by editing the fetcher. |
| `lib/yahoo_finance/valuation.py`, `statement_ordering.py`, `charting.py` | Pure computation, called unchanged. |
| `lib/yahoo_finance/answer.py` | Untouched **unless** the streamed-answer option in `06-open-decisions.md` is chosen — then a new sibling `generate_answer_streamed` is added; `generate_answer` itself still isn't modified. |
| `public/js/markdown.js`, `math.js`, `chart.js` | The answer still renders through these exactly as now. |
| `public/vercel.json` | The `/api/*` rewrite already covers the new route. Only revisit it if Vercel won't stream (see `02-transport.md`). |

## Known risks, ranked by how likely they are to bite

1. **A proxy buffers the stream and it arrives all at once.** Render's
   front proxy, or the Vercel rewrite, may hold the response until it
   completes — defeating the whole feature while looking like it "works" on
   localhost. Mitigations: `X-Accel-Buffering: no` + `Cache-Control:
   no-cache`; **test against the deployed URL and through the Vercel proxy,
   not just locally**; keep the frontend fallback so a buffered stream still
   yields a correct (if un-live) answer. If the Vercel proxy specifically is
   the problem, point `STREAM_BASE` at the Render URL directly.

2. **gunicorn worker exhaustion.** Each open stream pins one worker for the
   full answer duration (tens of seconds). With the default **sync** worker
   and a small worker count, a handful of concurrent users saturates the
   service and *new `/api/ask` requests also start queuing*. Mitigation:
   `--worker-class gthread --threads 8` (or `gevent`), raise `--timeout`
   past the longest answer, and consider a hard cap on concurrent streams
   that returns 503 (frontend then falls back).

3. **Something sensitive leaks into `detail`.** A well-meaning "let's also
   show the fetched revenue figure" turns the panel into a data-exfil
   surface, or a debugging `detail["prompt"] = ...` ships to prod.
   Mitigation: `01-progress-events.md` defines `detail` as a strict
   allowlist of routing metadata (tickers, module names, counts); the Step 2
   verification explicitly requires reading every `detail` payload; a small
   unit test asserting no `step` event's `detail`, JSON-serialised, contains
   the OpenAI key or any value from the `info` dict.

4. **Frontend coupling creeps back in.** Someone hard-codes stage labels in
   `render.js`, or infers "we must be fetching now" from a timer. Mitigation:
   labels come only from `event.label`; the panel is a generic list keyed by
   `stage`; a new backend stage must render with zero frontend changes (test
   this by adding a throwaway stage server-side).

5. **`localStorage` growth** (only if persistence is chosen). Every message
   gains a `steps` array with `detail` objects. 50 conversations × many
   messages × ~1 KB of steps adds up; `saveConversations` already trims to
   `MAX_CONVERSATIONS` but not per-message size. Mitigation: cap `detail` to
   a few short fields before saving (drop `reason`, keep name-lists and
   counts), and/or store only `{stage, status, label}` for finished
   messages, dropping `detail` on save.

6. **Agents SDK streaming API churn** (only if the streamed-answer option is
   chosen). `Runner.run_streamed`, `stream_events()`, the `event.type`
   strings, `response.output_text.delta` — young, moving API, same caveat the
   migration spec raises about `set_default_openai_client` et al. Mitigation:
   the coarse `answer` `running`/`done` events don't need any of it; treat
   token streaming as an enhancement that degrades to the sync
   `generate_answer` on any import/attribute error, logged once.

7. **Error semantics drift.** A mid-stream failure must still land in
   `message.error` and render through the existing `.answer.error` path — not
   as a broken panel or a silent stop. Mitigation: the stream's `error`
   event reuses the `{"error": "..."}` shape `/api/ask` already returns, and
   `onError` sets the same `message.error` field `askOnce` sets.

8. **Double work on fallback.** If `tryStream` fails *after* the pipeline
   already started server-side, and the frontend then calls `/api/ask`, the
   question runs twice (wasted OpenAI + Yahoo calls, ~2× latency).
   Mitigation: only fall back when the stream fails *before* any `step`
   arrived (i.e. the endpoint was unreachable / 404 / no body); once steps
   are flowing, a drop becomes a visible error, not a silent retry.

## The permission boundary, restated

This spec lives in `docs/thinking-process/` because most of the files it
changes are inside `lib/` and `public/`, which `.claude/CLAUDE.md` marks
write-off-limits for Claude. `api/ask.py` is the exception — that edit Claude
can make directly. Everything under `lib/` and `public/` must be applied by
you, or by a session explicitly told the restriction is lifted for the task.
Claude will not infer that permission from this file existing.
