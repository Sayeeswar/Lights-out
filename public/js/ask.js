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
    charts: [],
    steps: [],
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
      message.charts = data.charts || [];
      // Status + timing per pipeline stage; rendered as the "thinking"
      // panel. Absent on an older backend -> panel simply isn't drawn.
      message.steps = Array.isArray(data.steps) ? data.steps : [];
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
