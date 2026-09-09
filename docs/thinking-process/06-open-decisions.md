# Open decisions — "let's think about it afterwards"

Three choices are intentionally unmade. The rest of the spec is written so
that whichever way each lands, only a small, named part changes. Decide these
after Step 2 of `04-testing-and-rollout.md`, when the machinery is proven and
you can actually feel the tradeoffs.

For each: the question, the options, a recommendation, what depends on it,
and a quick way to decide.

---

## Decision 1 — the transport mechanism

**Question:** how does the backend push progress to one browser during one
question?

| Option | How it works | Pros | Cons |
|---|---|---|---|
| **SSE** (recommended) | one long-lived `text/event-stream` response; server writes `event:`/`data:` frames | standard, simple framing, natural fit for "answer streams too", one connection | pins a gunicorn worker for the whole answer; proxy buffering risk |
| **NDJSON** | one long response, one JSON object per line | trivial to emit (`json.dumps(x)+"\n"`), no frame syntax | no standard client; same worker/buffer caveats as SSE; marginal gain |
| **Job + poll** | `POST` returns a `job_id`; browser polls `GET /status/<id>` every ~1s | friendliest to proxies and to the sync worker; each request is short | in-process job state needs one worker or Redis; chattier; answer can't stream smoothly |

**Recommendation: SSE.** It's the least surprising choice, `02-transport.md`
already specs it end to end, and it's the only one where the "stream the
answer text" depth option (Decision 2) works well. Take Job+poll only if
Step 2 testing shows you cannot defeat proxy buffering on Render/Vercel *and*
you can't point the frontend at the Render URL directly.

**What depends on it:** `api/ask.py`'s new route, and the transport-reading
half of `public/js/thinking.js` + `ask.js`'s `tryStream`. The event model
(`01`) and the panel rendering (`03`) are identical for all three.

**How to decide quickly:** do Step 2 with SSE. Run the `curl -N` against the
**deployed** Render URL and the **Vercel-proxied** path. If both show frames
arriving one at a time → SSE, done. If Vercel buffers but Render doesn't →
SSE with `STREAM_BASE` = Render URL. If Render itself buffers and worker
tuning doesn't fix it → switch to Job+poll.

---

## Decision 2 — how much detail the panel shows

**Question:** what goes in the thinking panel?

| Option | Shows | Effort | Risk surface |
|---|---|---|---|
| **Labels only** | the 7 stage rows, spinner → check | smallest | none — no `detail` on the wire at all |
| **Labels + key facts** (recommended) | rows + the safe `detail` summary: detected ticker/modules, modules fetched + field counts, metric names, chart count | small | low — `detail` is the `01` allowlist |
| **Labels + facts + streamed answer** | the above, plus the answer text types in token-by-token under the panel | largest | adds SDK-streaming fragility (`01`, risk 6) |

**Recommendation: Labels + key facts.** It's the point of the feature —
users see *what the app decided and pulled*, not just that it's busy — and it
carries only routing metadata. Add streamed answer later as a pure
enhancement if it feels worth it; it changes feel, not information.

**What depends on it:**
- Labels only → `detail` can be dropped from events entirely; `renderStepDetail`
  in `03` isn't needed; no `answer_delta`.
- + facts → as specced in `01`/`03`.
- + streamed answer → also need `generate_answer_streamed` (`01`, last
  section), the `answer_delta` event, `onDelta` in `thinking.js`, and the
  Agents SDK migration must be done.

**How to decide quickly:** ship "Labels + key facts" to yourself for a day of
real use. If you keep wishing the answer appeared sooner, add streaming. If
the `detail` line feels like noise, fall back to "Labels only" — it's a
one-line render change.

---

## Decision 3 — persistence in chat history

**Question:** when a past conversation is reopened from Chat History, what
does the panel show?

| Option | Behaviour | Storage cost |
|---|---|---|
| **Save, collapsed & replayable** (recommended) | finished `steps` are stored with the message; history shows a collapsed `Thinking · N steps` that expands on click | ~0.3–1 KB per message |
| **Live only, discard after** | panel exists only while answering; once done it's gone; reopened history looks exactly like today (question + answer) | zero |

**Recommendation: Save, collapsed.** The panel's value isn't only "it's
working now" — "which modules did it pull for this answer?" is useful when
re-reading an old answer. Collapsed by default keeps the history view calm.

**What depends on it:**
- Save → keep `m.steps` in the object `saveConversations` writes; add the
  `loadConversations` repair line from `03`; apply risk-5 mitigation (trim
  `detail`, or store only `{stage,status,label}` for finished messages).
- Discard → clear `m.steps = []` in the `finally` block of `askQuestion`
  before `saveConversations()`; `storage.js` needs no change.

**How to decide quickly:** go with Save + trimmed detail. Only reconsider if
Step 3 testing shows `localStorage` quota errors in `saveConversations`'s
`catch` (unlikely at `MAX_CONVERSATIONS = 50`).

---

## Summary — the safe default set

If you just want a recommendation to run with:

1. **SSE**, with a deployed-path buffering test as the gate.
2. **Labels + key facts** (the `01` allowlist), no token streaming yet.
3. **Save, collapsed**, with `detail` trimmed on save.

This set needs: `pipeline.py` (`on_progress`), `api/ask.py` (SSE route),
`public/js/{thinking.js,ask.js,render.js,storage.js,config.js}`,
`index.html`, `style.css`. It does **not** need `answer.py` changes and does
**not** hard-depend on the Agents SDK migration (though it should still be
sequenced after it).
