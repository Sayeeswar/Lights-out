"""
Vercel entrypoint.

Vercel's Python runtime auto-detects a WSGI `app` object exported
from a file under /api and serves it as a serverless function.
This file becomes reachable at: POST https://<your-app>.vercel.app/api/ask
"""

import os
import sys
import traceback

from flask import Flask, request, jsonify

# Make /lib importable when this file runs as a standalone Vercel function
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.yahoo_finance import ask_stock_ai  # noqa: E402

app = Flask(__name__)


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        result = ask_stock_ai(question)
        return jsonify(result), 200

    except Exception as e:
        # Keep the error shape consistent with the original CLI's
        # `Error: {type}: {message}` behaviour, just as JSON.
        return jsonify({
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc() if os.getenv("DEBUG") else None,
        }), 500


@app.route("/api/ask", methods=["GET"])
def health():
    return jsonify({"status": "ok", "usage": "POST { \"question\": \"...\" } to this endpoint"}), 200
