# Frontend: the thinking panel

**Files this describes:** `public/js/ask.js`, `public/js/render.js`,
`public/js/storage.js`, a new `public/js/thinking.js`, `public/index.html`
(one `<script>` line), `public/style.css` (new classes).

**Every file here is inside `public/`, which is write-off-limits for Claude.**
This doc is the spec; you apply it. See rule 5 in `00-overview.md`.

## The decoupling rule, made concrete

The frontend does **not** know the pipeline. It does not contain the string
"Fetching Yahoo Finance data" — that text arrives in an event's `label`. It
does not decide a stage is done — it reads `status`. It has no API key, no
prompt, no business logic. Its entire job here is: read events, keep a list,
draw the list, and when the stream ends, render the answer exactly the way it
does today.

If the backend one day sends a new stage `sentiment`, the frontend should
show it with its backend-provided label **without a frontend change**. Build
it generic.

## Message state — new fields

`storage.js`'s header comment documents a message as:

```
message: { id, question, pending, error, answerHtml, answer, ticker, submodule, chart }
```

Add two fields:

```
message: { ..., steps: [step], streaming: bool }
  step:  { stage, status, label, detail }   // detail kept as-is from the event; may be {}
```

- `steps` — the ordered list of the latest state of each stage. Not one entry
  per event: when a `done` event arrives for a stage already in the list as
  `running`, **replace** it, don't append. Key by `stage`.
- `streaming` — `true` from the moment the stream opens until `done`/`error`.
  Distinct from `pending` (which stays, so old code paths keep working).

Bump the stored-schema handling: `loadConversations()` in `storage.js`
already repairs half-written messages. Add there:

```js
m.steps = Array.isArray(m.steps) ? m.steps : [];
if (m.streaming) m.streaming = false;   // a stream that never finished — treat like the pending repair
```

so conversations saved by the current build load without `undefined`s.

## New module: `js/thinking.js`

One job: given a `Response` whose body is the SSE stream, parse it and call
back into the app as events arrive.

```js
// js/thinking.js
//
// Reads a text/event-stream response body and dispatches each event.
// EventSource can't be used here because it is GET-only and we POST a body,
// so we read the raw stream and split on the blank-line frame delimiter.

async function readEventStream(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      if (frame.startsWith(":")) continue;           // heartbeat comment

      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;

      let payload;
      try { payload = JSON.parse(dataLines.join("\n")); }
      catch { continue; }                             // ignore an unparseable frame

      if (event === "step" && handlers.onStep) handlers.onStep(payload);
      else if (event === "answer_delta" && handlers.onDelta) handlers.onDelta(payload.text || "");
      else if (event === "done" && handlers.onDone) handlers.onDone(payload);
      else if (event === "error" && handlers.onError) handlers.onError(payload);
    }
  }
}
```

> **Learn-by-doing opportunity (later).** `onStep` needs a merge rule:
> upsert-by-`stage`, and decide what happens if a `done` arrives for a stage
> that was never `running` (out-of-order), or a second `done` for a stage.
> That merge is ~6 lines of genuine interface design — a good piece to write
> yourself when you implement this. Constraints to weigh: keep `steps` in
> pipeline order (use the `seq` field), never show a stage going
> `done` → `running`, and make it idempotent so a reconnect that replays
> events doesn't duplicate rows.

Add the script to `index.html` **before** `ask.js`:

```html
<script src="js/thinking.js"></script>
<script src="js/ask.js"></script>
```

## `ask.js` — try the stream, fall back cleanly

The current `askQuestion` does one `fetch(API_BASE + "/api/ask")` and awaits
`res.json()`. Restructure to:

```js
async function askQuestion(question) {
  // ... unchanged: build `message`, push it, render() ...

  const useStream = USE_STREAM;                 // new const in config.js, default true

  try {
    if (useStream && await tryStream(question, message)) {
      // tryStream resolved true => it handled everything (steps + final answer)
    } else {
      await askOnce(question, message);          // the existing /api/ask path, unchanged
    }
  } catch (err) {
    message.error = "Request failed: " + String(err);
  } finally {
    // ... unchanged: pending=false, streaming=false, save, setBusy(false), render() ...
  }
}
```

- `askOnce` is today's body, moved verbatim into its own function. Nothing
  about the non-streaming path changes.
- `tryStream`:
  - `fetch(STREAM_BASE + "/api/ask/stream", {method:"POST", ...})`.
  - If `!res.ok` or `res.status === 404` (flag off) or no `res.body` →
    `return false` so the caller falls back. No user-visible error.
  - Otherwise `message.streaming = true`, then
    `await readEventStream(res, { onStep, onDelta, onDone, onError })` where:
    - `onStep(ev)` — upsert into `message.steps` (the merge rule above),
      `render()`.
    - `onDelta(text)` — append to a `message.answerStreaming` buffer,
      `render()` (only if you chose the streamed-answer depth — see
      `06-open-decisions.md`).
    - `onDone(result)` — set `message.answerHtml`, `message.answer`,
      `message.ticker`, `message.submodule`, `message.charts` from `result`
      exactly as `askOnce` does today; `message.streaming = false`.
    - `onError(payload)` — `message.error = payload.error` (same field
      `askOnce` sets), `message.streaming = false`.
  - `return true` once the stream closed with a `done` or `error`.
  - Any thrown error inside `tryStream` after steps have already shown:
    if `message.answerHtml` is still empty, `return false` to fall back;
    if we already have partial content, set `message.error` about a dropped
    connection. Your call — document which you chose.

`config.js` gains:

```js
const USE_STREAM = true;                       // flip to false to disable the thinking panel
const STREAM_BASE = API_BASE;                  // or the Render URL directly if Vercel won't stream (see 02-transport.md)
```

## `render.js` — draw the panel

In `renderMessages`, for each message, when `m.steps && m.steps.length`,
render a `.thinking` block **above** `.answer` (and above `.loading` while
pending). Keep the existing `m.pending` / `m.error` / answer branches; the
panel is additive.

```js
function renderThinking(m) {
  if (!m.steps || !m.steps.length) return "";
  const live = m.streaming;
  const rows = m.steps.map((s) => `
    <li class="thinking-row is-${s.status}">
      <span class="thinking-dot" aria-hidden="true"></span>
      <span class="thinking-label">${escapeHtml(s.label)}</span>
      ${renderStepDetail(s)}
    </li>`).join("");

  // Live: always expanded, aria-live so a screen reader announces each step.
  // Finished: a collapsed <details> summarising "Thinking (N steps)".
  if (live) {
    return `<div class="thinking" role="status" aria-live="polite">
              <ul class="thinking-list">${rows}</ul>
            </div>`;
  }
  return `<details class="thinking is-collapsed">
            <summary>Thinking · ${m.steps.length} steps</summary>
            <ul class="thinking-list">${rows}</ul>
          </details>`;
}

function renderStepDetail(s) {
  const d = s.detail || {};
  // Generic: show a few known-safe keys if present, else nothing.
  // NOTE: labels/keys here are display sugar only — never logic.
  const bits = [];
  if (Array.isArray(d.ticker) && d.ticker.length) bits.push(d.ticker.join(", "));
  if (Array.isArray(d.module)) bits.push(d.module.join(" · "));
  if (Array.isArray(d.modules)) bits.push(d.modules.join(" · "));
  if (Array.isArray(d.metrics)) bits.push(d.metrics.join(", "));
  if (typeof d.count === "number") bits.push(d.count + " chart" + (d.count === 1 ? "" : "s"));
  return bits.length ? `<span class="thinking-detail">${escapeHtml(bits.join("  —  "))}</span>` : "";
}
```

Wire it into the entry HTML the same way `chartBoxes` is:

```js
entry.innerHTML = `
  <div class="question">${escapeHtml(m.question)}</div>
  ${renderThinking(m)}
  <div class="meta">${tickerTags}${submoduleTags}</div>
  <div class="answer">${answerBody}</div>
  ${chartBoxes}
`;
```

While `m.streaming` and no answer yet, `answerBody` is the streamed buffer
(if that option was chosen) or the existing `<span class="spinner">` markup.

## `style.css` — classes to add

Off-limits for Claude to edit; here's the list so you can style them to match
the existing IBM Plex / muted palette:

| class | purpose |
|---|---|
| `.thinking` | the panel container; subtle left border or tinted background |
| `.thinking-list` / `.thinking-row` | the checklist; `list-style:none`, tight rows |
| `.thinking-row.is-running` | current step — animated dot, slightly bolder |
| `.thinking-row.is-done` | done — check-mark dot, muted text |
| `.thinking-row.is-skipped` | skipped — dash dot, most muted |
| `.thinking-row.is-error` | error — warning colour |
| `.thinking-dot` | the leading status glyph |
| `.thinking-label` / `.thinking-detail` | text; detail is smaller + mono |
| `details.thinking > summary` | the collapsed "Thinking · N steps" affordance |

Respect `@media (prefers-reduced-motion: reduce)` — no pulsing dot.

## Accessibility

- Live panel: `role="status" aria-live="polite"` so each new step is
  announced without stealing focus.
- Collapsed panel: a real `<details>/<summary>` so it's keyboard-operable for
  free.
- The panel is supplementary — the answer must be fully usable with the panel
  collapsed or ignored.

## What must NOT change

- The non-streaming path (`askOnce`) — identical requests, identical
  rendering when `steps` is empty.
- The existing message fields and the `localStorage` key/shape for them.
  `steps`/`streaming` are additive; old saved messages (no `steps`) render
  exactly as before.
- Chart rendering, math typesetting, the history/current toggle, titles.
- `escapeHtml` on every piece of event text — `label` and `detail` come from
  the backend but still go through the same escaping as `question`.

## How to verify

Browser, against a backend with `ENABLE_PROGRESS_STREAM=1`:

1. Ask a valuation question. The panel appears, rows flip `running → done`
   over several seconds, `intent_adjust` shows as done ("Added company
   info…"), `charts` shows "No charts for this question", then the answer
   renders — same content, table, LaTeX, bold numbers as before.
2. Ask a history question. `reshape`/`valuation` show as **skipped**; a chart
   renders.
3. Set `USE_STREAM = false` (or `ENABLE_PROGRESS_STREAM` unset). App behaves
   exactly as today — single "Thinking…" spinner, no panel, no console
   errors.
4. Kill the backend mid-answer. The frontend either falls back to `/api/ask`
   or shows the dropped-connection error you chose — never a stuck spinner.
5. Reopen the conversation from Chat History and confirm it matches whatever
   `06-open-decisions.md` decided for persistence.
