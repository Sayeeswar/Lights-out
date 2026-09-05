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
