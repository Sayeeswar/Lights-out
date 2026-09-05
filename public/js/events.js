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
