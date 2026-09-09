# Backend: getting events to the browser

**File this describes:** `api/ask.py` (a new route). This file is **not**
inside `lib/` or `public/`, so this is an edit Claude can make directly.
**Also relevant:** `public/vercel.json` (the rewrite that proxies `/api/*`),
and the gunicorn command in Render's config.

## The endpoint strategy

Leave `POST /api/ask` **exactly** as it is. Add a **new** route for the
streamed version. The frontend tries the new one and falls back to the old
one (see `03-frontend.md`), so the old route is both the compatibility
guarantee and the safety net.

Provisional name: `POST /api/ask/stream`. Same request body as `/api/ask`
(`{"question": "..."}`), same auth (none), same CORS.

> The **wire format** of this endpoint — SSE vs newline-delimited JSON vs a
> job-id you poll — is deliberately **not decided yet**. See
> `06-open-decisions.md`. The rest of this file describes the **recommended**
> option (SSE) in full, and sketches the other two so the decision has
> something concrete to compare. If you pick a different option, only this
> file and the transport-reading part of `03-frontend.md` change; the event
> model (`01`) and the panel UI (`03`) are the same either way.

---

## Recommended: Server-Sent Events (SSE)

One long-lived HTTP response with `Content-Type: text/event-stream`. The
server writes `event:`/`data:` blocks as the pipeline progresses and closes
the response when done.

### The wire contract

Four event names:

```
event: step
data: {"stage":"intent","status":"running","label":"Working out what data your question needs","detail":{},"seq":1,"ts":...}

event: step
data: {"stage":"intent","status":"done","label":"Identified the data to fetch","detail":{"ticker":["AAPL"],"module":["info"],"submodule":[],"reason":"..."},"seq":2,"ts":...}

: heartbeat            <- a bare comment line every ~15s so idle proxies don't close the connection

event: answer_delta    <- only if the "stream the answer" option is chosen
data: {"text":"Apple's trailing P/E"}

event: done
data: {"question":"...","intent":{...},"answer":"...","answer_html":"...","charts":[...]}

event: error
data: {"error":"RuntimeError: ...","stage":"fetch"}
```

- Every `step` `data` payload is exactly one progress event from
  `01-progress-events.md`.
- `done` carries the **exact dict `ask_stock_ai` returns today** — the
  frontend renders the final answer from this, not from the deltas.
- Exactly one of `done` or `error` is sent, and it is always last.
- After `done`/`error` the server closes the stream.

### Flask implementation shape

```python
# api/ask.py

import json, queue, threading
from flask import Response, stream_with_context

def _sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

@app.route("/api/ask/stream", methods=["POST"])
def ask_stream():
    if os.getenv("ENABLE_PROGRESS_STREAM") != "1":
        return jsonify({"error": "streaming disabled"}), 404   # feature flag

    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # The pipeline is synchronous and calls on_progress from its own thread of
    # execution; a thread-safe queue bridges it to the response generator.
    events = queue.Queue()
    SENTINEL = object()

    def run():
        try:
            result = ask_stock_ai(question, on_progress=events.put)
            events.put(("done", result))
        except Exception as e:
            events.put(("error", {"error": f"{type(e).__name__}: {e}"}))
        finally:
            events.put(SENTINEL)

    threading.Thread(target=run, daemon=True).start()

    @stream_with_context
    def generate():
        while True:
            item = events.get()
            if item is SENTINEL:
                return
            if isinstance(item, dict):                 # a progress event
                yield _sse("step", item)
            else:                                       # ("done"|"error", payload)
                yield _sse(item[0], item[1])

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"           # ask nginx-style proxies not to buffer
    resp.headers["Connection"] = "keep-alive"
    return resp
```

Notes:

- `on_progress=events.put` works because `queue.Queue.put` matches
  `Callable[[dict], None]`. The `_emit` wrapper in `01` already swallows
  exceptions, so a full queue can't crash the pipeline.
- A background thread runs the pipeline so the response generator can stream
  while `ask_stock_ai` is still working. If you took the async
  `generate_answer_streamed` option, that thread wraps the answer stage in
  `asyncio.run(...)` and pushes `answer_delta` events onto the same queue.
- Add the heartbeat by using `events.get(timeout=15)` and yielding
  `": heartbeat\n\n"` on `queue.Empty`.
- CORS: `flask_cors` is already configured for `r"/*"`, so the new route is
  covered. Verify the preflight passes for `POST` with `Content-Type:
  application/json`.

### Hosting-platform caveats (these are the real risks)

1. **gunicorn worker model.** One SSE request occupies one worker for the
   full duration of the answer (10–40s+). With the default **sync** worker
   and, say, 2 workers, three simultaneous questions and you're out of
   capacity. Move to a threaded or async worker for this to be safe:

   ```
   gunicorn api.ask:app --worker-class gthread --threads 8 --timeout 120 --bind 0.0.0.0:$PORT
   ```

   (or `--worker-class gevent`). Also raise `--timeout` above the longest
   expected answer, or gunicorn kills the worker mid-stream.

2. **Render's proxy may buffer.** Render sits a proxy in front of your
   service. `X-Accel-Buffering: no` and `Cache-Control: no-cache` tell
   well-behaved proxies to pass bytes through immediately; they are not a
   guarantee. **Test through the deployed URL**, not just localhost — if the
   whole stream arrives in one burst at the end, buffering is the culprit and
   the frontend fallback (which still works) is what users get.

3. **Vercel rewrite.** `public/vercel.json` rewrites `/api/:path*` to the
   Render service. Vercel rewrites proxy the upstream response through; SSE
   generally survives this, but confirm on a deployed preview. If it does not
   stream through Vercel, the frontend can call the Render URL directly for
   the stream endpoint (it already knows it — `js/config.js` has
   `https://yahoo-finance-ai-api.onrender.com`), keeping same-origin only for
   the non-streaming fallback.

4. **Client disconnect.** When the user navigates away, the generator gets
   `GeneratorExit`. The daemon background thread keeps running to completion
   (harmless, it just fills a queue nobody drains). If you want to cancel the
   pipeline on disconnect, that's a larger change — out of scope here.

---

## Alternative A: newline-delimited JSON (NDJSON)

Same idea, blunter framing. `Content-Type: application/x-ndjson`, body is one
JSON object per line:

```
{"type":"step","stage":"intent","status":"running",...}
{"type":"step","stage":"fetch","status":"running",...}
{"type":"done","question":"...","answer_html":"...",...}
```

- Pro: no `event:`/`data:` ceremony; `json.dumps(obj) + "\n"` per line.
- Con: no standard client object (SSE has `EventSource`, though we can't use
  it here anyway — see `03`); you hand-parse lines either way, so the
  practical difference is small.
- Same worker/buffering caveats as SSE.

## Alternative B: job + poll

`POST /api/ask` returns `{"job_id": "..."}` immediately; the pipeline runs in
a background thread writing progress into an in-process dict keyed by job id;
the frontend polls `GET /api/ask/status/<job_id>` every ~1s and gets the
steps-so-far plus, eventually, the final result.

- Pro: no long-lived connections — friendliest to every proxy and to the
  sync gunicorn worker (each poll is a fast request).
- Con: in-process job state doesn't survive a worker restart and isn't shared
  across workers (a poll can hit a different worker than the one running the
  job) — you'd need a single worker, or Redis, or sticky routing. Chattier.
  The answer can't stream smoothly (you get it in ~1s-granular chunks or all
  at once).
- Changes `POST /api/ask`'s response shape **only if** you overload the same
  route; better to add `POST /api/ask/start` and leave `/api/ask` alone.

See `06-open-decisions.md` for how to choose.

## What must NOT change

- `POST /api/ask` — request, response, status codes, headers. Untouched.
- The `error` JSON shape the frontend already knows (`{"error": "..."}`) —
  the stream's `error` event reuses it so `ask.js` error handling is shared.
- `GET /`, `GET /healthz`, `GET /api/ask` (the usage stub) — untouched.

## How to verify

```
# streaming, deployed or local (note -N = no curl buffering):
! curl -N -X POST "$BASE/api/ask/stream" -H "Content-Type: application/json" -d '{"question":"Is Microsoft overvalued?"}'
```

You should see `event: step` lines appear **one at a time over several
seconds**, then a single `event: done`. If they all appear together at the
end, a buffer somewhere is defeating the point — check worker class, then
`X-Accel-Buffering`, then test the Render URL directly to isolate Vercel.
