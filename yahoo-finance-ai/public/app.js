// API base. When the page is opened from localhost we assume a local backend
// (e.g. `python -m flask --app api/ask run --port 3000`); otherwise we call
// the deployed Render service directly.
const LOCAL_HOSTS = ["localhost", "127.0.0.1"];
const API_BASE = LOCAL_HOSTS.includes(location.hostname)
  ? "http://localhost:3000"
  : "https://yahoo-finance-ai-api.onrender.com";

const input = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const historyEl = document.getElementById("history");
const examplesEl = document.getElementById("examples");
const viewToggleBtn = document.getElementById("view-toggle");
const newChatBtn = document.getElementById("new-chat");

// ----------------------------------------------------------------------------
// Conversation store
//
// One question + its answer is a "message"; a run of messages is a
// "conversation". Every conversation is kept in localStorage. The toggle
// button only changes which conversation(s) are on screen — switching views
// never deletes anything.
//
//   conversation: { id, title, createdAt, updatedAt, messages: [message] }
//   message:      { id, question, pending, error, answerHtml, answer,
//                   ticker, submodule, chart }
// ----------------------------------------------------------------------------
const STORAGE_KEY = "yfa.conversations";
const MAX_CONVERSATIONS = 50;
const TITLE_MAX = 60;

let conversations = [];
let activeId = null;
let viewMode = "current"; // "current" = active conversation only, "history" = all
let openedId = null; // a conversation opened from the history list
let chartSeq = 0;

function uid() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function loadConversations() {
  let parsed = [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const value = JSON.parse(raw);
      if (Array.isArray(value)) parsed = value;
    }
  } catch (err) {
    parsed = [];
  }

  // Repair anything half-written (e.g. the tab was closed mid-request) so the
  // UI never shows a permanent spinner.
  parsed.forEach((conv) => {
    conv.messages = Array.isArray(conv.messages) ? conv.messages : [];
    conv.title = conv.title || "";
    conv.createdAt = conv.createdAt || Date.now();
    conv.updatedAt = conv.updatedAt || conv.createdAt;
    conv.messages.forEach((m) => {
      if (m.pending) {
        m.pending = false;
        if (!m.error && !m.answerHtml && !m.answer) {
          m.error = "This answer wasn't saved (the page closed while it was loading).";
        }
      }
    });
  });

  return parsed;
}

function saveConversations() {
  try {
    if (conversations.length > MAX_CONVERSATIONS) {
      const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
      const kept = sorted.slice(0, MAX_CONVERSATIONS);
      if (activeId && !kept.some((c) => c.id === activeId)) {
        const active = conversations.find((c) => c.id === activeId);
        if (active) kept[kept.length - 1] = active;
      }
      conversations = kept;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch (err) {
    /* storage full or blocked — carry on with in-memory state */
  }
}

function newConversation() {
  const now = Date.now();
  return { id: uid(), title: "", createdAt: now, updatedAt: now, messages: [] };
}

function getActive() {
  return conversations.find((c) => c.id === activeId) || null;
}

function ensureActive() {
  let active = getActive();
  if (!active) {
    active = newConversation();
    conversations.push(active);
    activeId = active.id;
  }
  return active;
}

function conversationTitle(conv) {
  if (conv.title) return conv.title;
  const first = conv.messages.find((m) => m.question);
  const text = ((first && first.question) || "New conversation").trim();
  return text.length > TITLE_MAX ? text.slice(0, TITLE_MAX - 1) + "…" : text;
}

// ----------------------------------------------------------------------------
// Asking a question
// ----------------------------------------------------------------------------
async function askQuestion(question) {
  question = (question || "").trim();
  if (!question) return;

  const conv = ensureActive();
  const message = {
    id: uid(),
    question,
    pending: true,
    error: null,
    answerHtml: "",
    answer: "",
    ticker: "",
    submodule: "",
    chart: null,
  };
  conv.messages.push(message);
  conv.updatedAt = Date.now();
  if (!conv.title) conv.title = conversationTitle(conv);
  saveConversations();

  // A new question always belongs to the current chat — show it there.
  viewMode = "current";
  openedId = null;
  setBusy(true);
  render();

  try {
    const res = await fetch(API_BASE + "/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    if (!res.ok) {
      message.error = data.error || "Something went wrong.";
    } else {
      const intent = data.intent || {};
      message.answerHtml = data.answer_html || "";
      message.answer = data.answer || "";
      message.ticker = intent.ticker || "";
      message.submodule = intent.submodule || "";
      message.chart = data.chart || null;
    }
  } catch (err) {
    message.error = "Request failed: " + String(err);
  } finally {
    message.pending = false;
    conv.updatedAt = Date.now();
    saveConversations();
    setBusy(false);
    render();
    input.value = "";
    input.focus();
  }
}

function setBusy(busy) {
  askBtn.disabled = busy;
  input.disabled = busy;
}

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
      let canvasId = "";
      if (m.chart) {
        canvasId = `chart-${++chartSeq}`;
        pendingCharts.push({ canvasId, chart: m.chart });
      }
      entry.innerHTML = `
        <div class="question">${escapeHtml(m.question)}</div>
        <div class="meta">
          ${m.ticker ? `<span class="badge">${escapeHtml(m.ticker)}</span>` : ""}
          ${m.submodule ? `<span class="badge">${escapeHtml(m.submodule)}</span>` : ""}
        </div>
        <div class="answer">${answerBody}</div>
        ${m.chart ? `<div class="chart-box"><canvas id="${canvasId}"></canvas></div>` : ""}
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

// Typeset any \( ... \) / \[ ... \] math the backend left in the answer.
// KaTeX only; we intentionally do NOT register $...$ so currency like
// "$5.2 billion" is never mistaken for math. Purely visual — a failure
// here must never blank out the answer text.
function typesetMath(el) {
  if (typeof renderMathInElement !== "function") return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  } catch (err) {
    /* leave the raw \( ... \) visible rather than breaking the card */
  }
}

// "1987-05-11T09:30:00-04:00" -> "1987-05". Anything that doesn't start with
// YYYY-MM is returned unchanged. Used for both the x-axis ticks and the
// hover tooltip title so neither shows the time part.
function monthYear(value) {
  const match = /^(\d{4})-(\d{2})/.exec(String(value));
  return match ? `${match[1]}-${match[2]}` : String(value);
}

function renderChart(canvasId, chart) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  new Chart(ctx, {
    type: chart.chart_type === "bar" ? "bar" : "line",
    data: {
      labels: chart.x,
      datasets: [
        {
          label: chart.label || "",
          data: chart.y,
          borderColor: "#4f7cff",
          backgroundColor: chart.chart_type === "bar"
            ? "rgba(79, 124, 255, 0.6)"
            : "rgba(79, 124, 255, 0.15)",
          fill: chart.chart_type !== "bar",
          tension: 0.25,
          pointRadius: chart.x.length > 60 ? 0 : 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: {
            color: "#8a90a0",
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 8,
            callback(value) {
              return monthYear(this.getLabelForValue(value));
            },
          },
          grid: { color: "#262b36" },
        },
        y: {
          ticks: { color: "#8a90a0" },
          grid: { color: "#262b36" },
        },
      },
      plugins: {
        legend: { labels: { color: "#e6e8ec" } },
        tooltip: {
          callbacks: {
            title(items) {
              return items.length ? monthYear(items[0].label) : "";
            },
          },
        },
      },
    },
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Minimal, safe Markdown -> HTML for the FALLBACK path only (when the backend
// didn't send pre-rendered answer_html). Every line is HTML-escaped first, and
// the only tags produced are the fixed set this function writes itself, so no
// markup from `src` can slip through. Handles: # headings, - / * / 1. lists,
// blank-line paragraphs, single newline -> <br>, **bold**, *italic*, `code`.
// Tables are NOT handled here — that needs the backend renderer.
function renderMarkdown(src) {
  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

  const lines = String(src).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let list = null; // "ul" | "ol" | null
  let para = [];

  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + para.map(inline).join("<br>") + "</p>");
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      out.push("</" + list + ">");
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");

    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      const level = heading[1].length;
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushPara();
      const want = bullet ? "ul" : "ol";
      if (list !== want) {
        flushList();
        list = want;
        out.push("<" + want + ">");
      }
      out.push("<li>" + inline((bullet || numbered)[1]) + "</li>");
      continue;
    }

    flushList();
    para.push(line);
  }

  flushPara();
  flushList();
  return out.join("\n");
}

// ----------------------------------------------------------------------------
// Wiring
// ----------------------------------------------------------------------------
askBtn.addEventListener("click", () => askQuestion(input.value));

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion(input.value);
});

examplesEl.addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    askQuestion(e.target.textContent);
  }
});

viewToggleBtn.addEventListener("click", () => {
  viewMode = viewMode === "history" ? "current" : "history";
  openedId = null;
  render();
});

newChatBtn.addEventListener("click", () => {
  const active = getActive();
  if (!active || active.messages.length > 0) {
    const conv = newConversation();
    conversations.push(conv);
    activeId = conv.id;
    saveConversations();
  }
  viewMode = "current";
  openedId = null;
  render();
  input.focus();
});

// ----------------------------------------------------------------------------
// Startup: resume the most recent conversation, or begin a fresh one.
// ----------------------------------------------------------------------------
conversations = loadConversations();
if (conversations.length > 0) {
  activeId = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
} else {
  const first = newConversation();
  conversations.push(first);
  activeId = first.id;
}
render();
