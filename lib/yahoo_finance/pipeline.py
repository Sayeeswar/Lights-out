"""
The complete question -> intent -> data -> answer pipeline.
"""

from lib.formatting import markdown_to_safe_html

from .answer import generate_answer
from .charting import build_chart_payload
from .fetch import execute_yahoo_intent
from .intent import detect_intent
from .statement_ordering import STATEMENT_SUBMODULES, reorder_statement_data


def ask_stock_ai(question: str) -> dict:
    """
    Runs the full pipeline and returns a dict suitable for
    a JSON HTTP response (rather than printing to a console).
    """

    intent = detect_intent(question)
    yahoo_data = execute_yahoo_intent(intent)

    for module, data in yahoo_data.items():
        if module in STATEMENT_SUBMODULES:
            annual = not module.startswith("quarterly_")
            yahoo_data[module] = reorder_statement_data(data, annual=annual)

    answer = generate_answer(question, intent, yahoo_data)
    charts = build_chart_payload(intent, yahoo_data)

    return {
        "question": question,
        "intent": intent,
        "answer": answer,  # raw Markdown, kept for API compatibility
        "answer_html": markdown_to_safe_html(answer),  # sanitized, UI-ready
        "charts": charts,  # [] when nothing in the intent is chartable
    }
