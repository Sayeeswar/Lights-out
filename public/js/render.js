// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------
function render() {
  updateToggleButton();

  if (viewMode === "history") {
    const opened = openedId ? conversations.find((c) => c.id === openedId) : null;
    if (openedId && !opened) openedId = null;
    if (!openedId) {
      renderConversationList();
      return;
    }
    renderMessages(opened, true);
    return;
  }

  renderMessages(getActive(), false);
}

function updateToggleButton() {
  const inHistory = viewMode === "history";
  viewToggleBtn.textContent = inHistory ? "🕘 Chat History" : "🗨 Current Chat Only";
  viewToggleBtn.setAttribute("aria-pressed", String(inHistory));
  viewToggleBtn.title = inHistory
    ? "Showing every saved conversation — click to show only the current chat"
    : "Showing only the current chat — click to browse all saved conversations";
}

function renderConversationList() {
  historyEl.innerHTML = "";

  const list = conversations
    .filter((c) => c.messages.length > 0)
    .sort((a, b) => b.updatedAt - a.updatedAt);

  if (list.length === 0) {
    historyEl.appendChild(makeNote("No conversations yet. Ask a question to start one."));
    return;
  }

  list.forEach((conv) => {
    const count = conv.messages.length;
    const when = new Date(conv.updatedAt).toLocaleString();
    const card = document.createElement("button");
    card.type = "button";
    card.className = "conv-card";
    card.innerHTML = `
      <span class="conv-title">${escapeHtml(conversationTitle(conv))}</span>
      <span class="conv-sub">
        ${conv.id === activeId ? `<span class="badge">Current</span>` : ""}
        ${count} message${count === 1 ? "" : "s"} · ${escapeHtml(when)}
      </span>
    `;
    card.addEventListener("click", () => {
      openedId = conv.id;
      render();
    });
    historyEl.appendChild(card);
  });
}

function renderMessages(conv, showBack) {
  historyEl.innerHTML = "";
  const pendingCharts = [];

  if (showBack) {
    const back = document.createElement("button");
    back.type = "button";
    back.className = "back-btn";
    back.textContent = "← All conversations";
    back.addEventListener("click", () => {
      openedId = null;
      render();
    });
    historyEl.appendChild(back);
  }

  const messages = (conv && conv.messages) || [];

  if (messages.length === 0) {
    if (showBack) historyEl.appendChild(makeNote("This conversation is empty."));
    return;
  }

  // Newest first, matching the original prepend order.
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    const entry = document.createElement("div");
    entry.className = "entry";

    if (m.pending) {
      entry.innerHTML = `
        <div class="question">${escapeHtml(m.question)}</div>
        <div class="loading"><span class="spinner"></span> Thinking…</div>
      `;
    } else if (m.error) {
      entry.innerHTML = `
        <div class="question">${escapeHtml(m.question)}</div>
        <div class="answer error">${escapeHtml(m.error)}</div>
      `;
    } else {
      // Preferred: answer_html — Markdown already rendered AND sanitized by
      // the backend. Fallback (older backend that only sends raw `answer`):
      // render a safe Markdown subset here so **bold**, bullets, headings and
      // paragraphs display instead of showing literal markers.
      const answerBody = m.answerHtml
        ? m.answerHtml
        : renderMarkdown(m.answer || "");
      // intent.ticker / intent.submodule can each be a list (e.g. two rows
      // being compared) - render one badge per entry rather than joining
      // them into a single string.
      const asTags = (v) => (Array.isArray(v) ? v : v ? [v] : []).filter(Boolean);
      const tickerTags = asTags(m.ticker)
        .map((t) => `<span class="badge">${escapeHtml(t)}</span>`)
        .join("");
      const submoduleTags = asTags(m.submodule)
        .map((s) => `<span class="badge">${escapeHtml(s)}</span>`)
        .join("");
      const charts = Array.isArray(m.charts) ? m.charts : [];
      const chartBoxes = charts
        .map((chart) => {
          const canvasId = `chart-${++chartSeq}`;
          pendingCharts.push({ canvasId, chart });
          return `<div class="chart-box"><canvas id="${canvasId}"></canvas></div>`;
        })
        .join("");
      entry.innerHTML = `
        <div class="question">${escapeHtml(m.question)}</div>
        <div class="meta">
          ${tickerTags}
          ${submoduleTags}
        </div>
        ${renderStepsPanel(m.steps)}
        <div class="answer">${answerBody}</div>
        ${chartBoxes}
      `;
      typesetMath(entry);
    }

    historyEl.appendChild(entry);
  }

  pendingCharts.forEach(({ canvasId, chart }) => renderChart(canvasId, chart));
}

function makeNote(text) {
  const p = document.createElement("p");
  p.className = "empty-note";
  p.textContent = text;
  return p;
}

// Retrospective "thinking" panel: the pipeline's per-stage status + timing,
// as sent in `data.steps`. Presentation only — the backend owns the labels,
// the order and the durations; this just lays them out. Rows fade in
// staggered via CSS (see `.step` in style.css) so it reads as progress even
// though the whole array arrived in one response.
function renderStepsPanel(steps) {
  if (!Array.isArray(steps) || steps.length === 0) return "";

  const rows = steps
    .map((s, i) => {
      const isError = s && s.status === "error";
      const mark = isError ? "✗" : "✓"; // ✗ / ✓
      const label = escapeHtml((s && (s.label || s.id)) || "");
      const time = escapeHtml(formatDuration((s && s.duration_ms) || 0));
      const msg =
        isError && s && s.error
          ? `<span class="step-msg">${escapeHtml(s.error)}</span>`
          : "";
      return `<li class="step${isError ? " step-error" : ""}" style="--i:${i}">
          <span class="step-mark">${mark}</span>
          <span class="step-label">${label}</span>
          ${msg}
          <span class="step-time">${time}</span>
        </li>`;
    })
    .join("");

  const n = steps.length;
  return `<details class="steps" open>
      <summary class="steps-summary">Thinking · ${n} step${n === 1 ? "" : "s"}</summary>
      <ul class="steps-list">${rows}</ul>
    </details>`;
}

// Format a millisecond count for a thinking-panel row: "<1ms" for an
// instant (or unmeasurable) step, "<n>ms" below one second, and seconds
// with two decimals above (e.g. "1.87s", "6.25s"). The `!(ms > 0)` guard
// also catches NaN / undefined / negatives.
function formatDuration(ms) {
  if (!(ms > 0)) return "<1ms";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
