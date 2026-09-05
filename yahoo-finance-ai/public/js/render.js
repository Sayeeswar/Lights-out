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

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
