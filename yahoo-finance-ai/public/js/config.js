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
