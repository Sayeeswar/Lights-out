# 🚀 How to Run the Yahoo Finance AI Assistant

This guide explains how to run the **Equity Research AI Assistant** locally and deploy it to production.

---

## 📋 Quick Start (Local Development)

### Prerequisites
- **Python 3.12** or higher
- **pip** (Python package manager)
- **OpenAI API key** (free or paid tier)

### Step 1: Clone & Navigate
```bash
git clone https://github.com/Sayeeswar/Lights-out.git
cd Lights-out/yahoo-finance-ai
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Copy the example file and add your OpenAI API key:
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
OPENAI_MODEL=gpt-4
```

> **Note:** Do NOT commit `.env` to Git — it's git-ignored for security.

### Step 4: Run the Flask Server
```bash
python -m flask --app api/ask run --port 3000
```

You should see:
```
 * Serving Flask app 'api/ask'
 * Running on http://127.0.0.1:3000
```

### Step 5: Test the API
In a new terminal, send a test request:
```bash
curl -X POST http://localhost:3000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Apple'\''s free cash flow trend?"}'
```

You'll get a JSON response with the answer and financial data.

### Step 6: Open the Web UI
Open your browser to:
```
http://localhost:3000
```

You should see a dark-themed chat interface. Ask questions like:
- "How much cash did Reliance make last year?"
- "What is Tesla's revenue growth?"
- "Show me Microsoft's balance sheet"

---

## 🌐 Production Deployment

The app is designed for a **split deployment**:
- **Backend (Python API)** → Render (free or paid)
- **Frontend (Static HTML/JS)** → Vercel (free tier works)

This split exists because the Python backend is too heavy (~500 MB with pandas/numpy) for Vercel's serverless 250 MB limit.

### Option A: Deploy on Render + Vercel (Recommended)

#### 1️⃣ Deploy Backend to Render

1. Push your repo to GitHub (if not already done):
   ```bash
   git push origin main
   ```

2. Go to [render.com](https://render.com) and sign in.

3. Click **New +** → **Blueprint**

4. Connect your GitHub repo and select the `Lights-out` repository.

5. Render will auto-detect `render.yaml` in the `yahoo-finance-ai/` folder.

6. When prompted, enter your **OPENAI_API_KEY** under environment variables:
   - **OPENAI_API_KEY**: Your actual key (do NOT put it in render.yaml)
   - **OPENAI_MODEL**: (optional) defaults to `gpt-4`
   - **CORS_ALLOW_ORIGIN**: (optional) restrict CORS if needed

7. Click **Deploy Blueprint**.

8. Wait 2-3 minutes for the build. Once live, note your service URL:
   ```
   https://yahoo-finance-ai-api.onrender.com
   ```

> **Free Tier Note:** The service spins down after ~15 min of inactivity. First request after idle takes 30–60s.

#### 2️⃣ Deploy Frontend to Vercel

1. Edit `yahoo-finance-ai/public/vercel.json`:
   ```json
   {
     "rewrites": [
       {
         "source": "/api/(.*)",
         "destination": "https://YOUR-RENDER-URL.onrender.com/api/$1"
       }
     ]
   }
   ```
   Replace `YOUR-RENDER-URL` with your actual Render service URL.

2. Go to [vercel.com](https://vercel.com) and sign in.

3. Click **Import Project** and select your GitHub repo.

4. Set **Root Directory** to `yahoo-finance-ai/public`.

5. Set **Framework Preset** to **Other** (no build command needed).

6. Click **Deploy**.

7. Your site is live at:
   ```
   https://<your-vercel-project>.vercel.app
   ```

---

### Option B: Run Both Locally (No Vercel/Render)

If you prefer a single-machine setup:

1. Start the Flask server (as per Quick Start above):
   ```bash
   cd yahoo-finance-ai
   python -m flask --app api/ask run --port 3000
   ```

2. In the same terminal or another window, serve the frontend:
   ```bash
   # Option 1: Python built-in server
   cd yahoo-finance-ai/public
   python -m http.server 8000
   ```
   Then open `http://localhost:8000`

   **OR**

   ```bash
   # Option 2: Use Python Flask to serve everything
   # (Modify api/ask.py to serve static files)
   ```

---

## 📝 API Endpoint Reference

### POST `/api/ask`

**Request:**
```json
{
  "question": "How much revenue did Apple make in 2023?"
}
```

**Response:**
```json
{
  "question": "How much revenue did Apple make in 2023?",
  "intent": {
    "ticker": "AAPL",
    "module": "ticker",
    "submodule": "info",
    "parameters": {},
    "reason": "User asked for Apple's revenue..."
  },
  "answer": "# Apple's Revenue\n\nApple generated **$383.3 billion** in revenue...",
  "answer_html": "<h1>Apple's Revenue</h1><p>Apple generated <strong>$383.3 billion</strong> in revenue...</p>"
}
```

**Fields:**
- `question` — Echo of your input
- `intent` — Parsed stock ticker, financial module, and submodule
- `answer` — Model's raw Markdown response
- `answer_html` — Rendered HTML (safe to inject into DOM)

---

## ⚙️ Environment Variables

| Variable | Required? | Default | Notes |
|----------|-----------|---------|-------|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI API key (sk-...) |
| `OPENAI_MODEL` | ❌ No | `gpt-4` | Model to use (gpt-4, gpt-3.5-turbo, etc.) |
| `CORS_ALLOW_ORIGIN` | ❌ No | `*` | Restrict CORS to specific origin if needed |

**Local Development:** Create `.env` in `yahoo-finance-ai/`
**Production (Render):** Set in Render dashboard → Environment
**Production (Vercel):** Not needed (key lives on Render)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'api'"
Make sure you're running from the `yahoo-finance-ai/` directory:
```bash
cd yahoo-finance-ai
python -m flask --app api/ask run
```

### "OPENAI_API_KEY not set"
1. Check `.env` exists in `yahoo-finance-ai/`
2. Verify the key is correctly pasted (no extra spaces)
3. Try: `python -c "import os; print(os.getenv('OPENAI_API_KEY'))"`

### 502 Bad Gateway / Service Unavailable (Render)
- Render free tier spins down after 15 min idle
- First request wakes it up (~30–60s wait)
- Check Render logs: Render dashboard → Logs tab

### "Connection refused" to backend (Vercel)
- Verify `public/vercel.json` has the correct Render URL
- Check CORS settings on Render (`CORS_ALLOW_ORIGIN`)
- Open browser DevTools → Network → check `/api/ask` request

### Slow yfinance responses
- No caching layer is implemented; each request fetches fresh data
- Consider adding Redis or Vercel KV for caching (optional improvement)

---

## 📂 File Guide

| File | Purpose |
|------|---------|
| `api/ask.py` | 🟢 Flask app & HTTP routes |
| `lib/yahoo_finance.py` | 🟢 Core logic: LLM + Yahoo Finance pipeline |
| `lib/formatting.py` | 🟢 Markdown → HTML converter |
| `public/index.html` | 🟢 Frontend UI (dark chat box) |
| `public/app.js` | 🟢 Frontend logic (fetch, KaTeX, Chart.js) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for local secrets |
| `wsgi.py` | Gunicorn entry point (Render) |
| `render.yaml` | Render deployment blueprint |
| `public/vercel.json` | Vercel config (proxies `/api/*` to Render) |

---

## 🎯 Next Steps

1. **Test locally** with the Quick Start guide above
2. **Deploy to Render** for the backend API
3. **Deploy to Vercel** for the frontend
4. **Customize** the UI in `public/index.html` and `public/app.js`
5. **(Optional) Add caching** to speed up repeated queries

---

## 📞 Support

- **Documentation:** See `README.md` for architecture & deployment details
- **Code Guide:** See `README.md` → File Guide for what each file does
- **Questions:** Refer to `questions.md` for common test queries

---

Happy analyzing! 📊
