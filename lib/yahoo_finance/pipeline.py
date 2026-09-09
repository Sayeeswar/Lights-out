"""
The complete question -> intent -> data -> answer pipeline.
"""

import time
from contextlib import contextmanager

from agents import trace

from lib.formatting import markdown_to_safe_html

from .answer import generate_answer
from .charting import build_chart_payload
from .fetch import execute_yahoo_intent
from .intent import detect_intent
from .statement_ordering import STATEMENT_SUBMODULES, reorder_statement_data
from .valuation import VALUATION_KEYWORDS, compute_valuation_metrics


def _wants_valuation(question: str, intent: dict) -> bool:
    """
    True when the question is about valuation multiples. Used to guarantee
    `info` (which carries price, market cap and the ratio fields) is
    fetched even when the intent router didn't list it.
    """
    haystack = f"{question} {intent.get('reason', '')}".lower()
    return any(keyword in haystack for keyword in VALUATION_KEYWORDS)


@contextmanager
def _step(steps: list, step_id: str, label: str):
    """
    Time one pipeline stage and append a status/timing record to `steps`.

    The record is only {id, label, status, duration_ms} -- never any data
    from inside the stage (no tickers, no fetched values, no answer text).
    On failure it records status "error" with a short message and then
    re-raises, so a stage that fails today still fails the request exactly
    as it does now.
    """
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        steps.append({
            "id": step_id,
            "label": label,
            "status": "error",
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "error": str(exc)[:200],
        })
        raise
    else:
        steps.append({
            "id": step_id,
            "label": label,
            "status": "done",
            "duration_ms": round((time.perf_counter() - started) * 1000),
        })


def ask_stock_ai(question: str) -> dict:
    """
    Runs the full pipeline and returns a dict suitable for
    a JSON HTTP response (rather than printing to a console).

    The whole run is wrapped in one Agents SDK trace, so the intent-router
    call, the fetch, and the answer call appear as child spans under a
    single "ask_stock_ai" trace in the OpenAI dashboard instead of two
    disconnected traces.

    The response also carries a `steps` array: one status/timing record per
    stage that actually ran ({id, label, status, duration_ms}), for the
    frontend "thinking" panel. It is additive -- a client that ignores
    `steps` sees exactly today's response.
    """

    steps: list[dict] = []

    with trace("ask_stock_ai", metadata={"question": question[:200]}):
        with _step(steps, "intent", "Detected intent"):
            intent = detect_intent(question)

        modules = intent.get("module") or []
        if _wants_valuation(question, intent) and "info" not in modules:
            with _step(steps, "intent_adjust", "Added valuation data"):
                modules.append("info")
                intent["module"] = modules

        with _step(steps, "fetch", "Fetched data"):
            yahoo_data = execute_yahoo_intent(intent)

        statement_modules = [m for m in yahoo_data if m in STATEMENT_SUBMODULES]
        if statement_modules:
            with _step(steps, "reshape", "Ordered financial statements"):
                for module in statement_modules:
                    annual = not module.startswith("quarterly_")
                    yahoo_data[module] = reorder_statement_data(
                        yahoo_data[module], annual=annual
                    )

        info = yahoo_data.get("info")
        if isinstance(info, dict) and info:
            with _step(steps, "valuation", "Computed valuation metrics"):
                yahoo_data["valuation_metrics"] = compute_valuation_metrics(info)

        with _step(steps, "answer", "Generated answer"):
            answer = generate_answer(question, intent, yahoo_data)

        with _step(steps, "charts", "Built charts"):
            charts = build_chart_payload(intent, yahoo_data)

        return {
            "question": question,
            "intent": intent,
            "answer": answer,  # raw Markdown, kept for API compatibility
            "answer_html": markdown_to_safe_html(answer),  # sanitized, UI-ready
            "charts": charts,  # [] when nothing in the intent is chartable
            "steps": steps,  # status + timing per stage; frontend "thinking" panel
        }
