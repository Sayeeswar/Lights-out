"""
HTTP API for the Equity Research AI assistant.

A plain WSGI (Flask) app. Deployed on Render:

    gunicorn api.ask:app --bind 0.0.0.0:$PORT

Run locally:

    flask --app api/ask run --port 3000

The frontend (hosted on Vercel) reaches this via a Vercel rewrite that
proxies /api/* to this service, so requests are same-origin in the browser.
"""

import os
import sys
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

# Make /lib importable regardless of the working directory gunicorn starts in.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.yahoo_finance import ask_stock_ai  # noqa: E402

app = Flask(__name__)

# Public, unauthenticated, read-only API. Defaults to allowing any origin;
# set CORS_ALLOW_ORIGIN to your frontend URL to lock it down.
CORS(app, resources={r"/*": {"origins": os.getenv("CORS_ALLOW_ORIGIN", "*")}})


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
        return jsonify({
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc() if os.getenv("DEBUG") else None,
        }), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ok",
        "service": "Equity Research AI Assistant API",
        "endpoints": {
            "POST /api/ask": "{ \"question\": \"...\" }",
            "GET /healthz": "liveness probe",
        },
    }), 200


@app.route("/api/ask", methods=["GET"])
def usage():
    return jsonify({
        "status": "ok",
        "usage": "POST { \"question\": \"...\" } to this endpoint",
    }), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    """Liveness probe for Render's health check."""
    return jsonify({"status": "ok"}), 200
