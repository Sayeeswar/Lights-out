"""
Core intent-detection + Yahoo Finance fetch + answer-generation logic.

Kept deliberately free of any HTTP/CLI concerns so it can be imported
by a serverless function (api/ask.py) without change.

Split by concern for easier debugging - see the individual modules:
config, json_safety, intent, fetch, statement_ordering, answer, charting,
pipeline. Only ask_stock_ai (the full pipeline) is re-exported here, since
that's the only symbol api/ask.py imports.
"""

from .pipeline import ask_stock_ai

__all__ = ["ask_stock_ai"]
