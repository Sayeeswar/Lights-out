"""
WSGI entry point.

Gunicorn / Render start command:  gunicorn wsgi:app
The real app lives in api/ask.py; this just re-exports it under the
conventional module name that host auto-detection looks for.
"""

from api.ask import app  # noqa: F401

if __name__ == "__main__":
    app.run(port=3000)
