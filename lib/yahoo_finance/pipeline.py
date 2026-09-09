"""
The complete question -> intent -> data -> answer pipeline.
"""

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


def ask_stock_ai(question: str) -> dict:
    """
    Runs the full pipeline and returns a dict suitable for
    a JSON HTTP response (rather than printing to a console).

    The whole run is wrapped in one Agents SDK trace, so the intent-router
    call, the fetch, and the answer call appear as child spans under a
    single "ask_stock_ai" trace in the OpenAI dashboard instead of two
    disconnected traces.
    """

    with trace("ask_stock_ai", metadata={"question": question[:200]}):
        intent = detect_intent(question)

        modules = intent.get("module") or []
        if _wants_valuation(question, intent) and "info" not in modules:
            modules.append("info")
            intent["module"] = modules

        yahoo_data = execute_yahoo_intent(intent)

        for module, data in yahoo_data.items():
            if module in STATEMENT_SUBMODULES:
                annual = not module.startswith("quarterly_")
                yahoo_data[module] = reorder_statement_data(data, annual=annual)

        info = yahoo_data.get("info")
        if isinstance(info, dict) and info:
            yahoo_data["valuation_metrics"] = compute_valuation_metrics(info)

        answer = generate_answer(question, intent, yahoo_data)
        charts = build_chart_payload(intent, yahoo_data)

        return {
            "question": question,
            "intent": intent,
            "answer": answer,  # raw Markdown, kept for API compatibility
            "answer_html": markdown_to_safe_html(answer),  # sanitized, UI-ready
            "charts": charts,  # [] when nothing in the intent is chartable
        }
