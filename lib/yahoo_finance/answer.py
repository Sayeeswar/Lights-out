"""
Generate the final Markdown answer from a question, its detected intent,
and the fetched Yahoo Finance data.

One LLM call, via the OpenAI Agents SDK. The role framing, the numbered
instructions and ANSWER_FORMATTING_RULES are baked into the Agent's
instructions once at import time; the question, intent and fetched data
vary per call and are passed as the run input. No output_type is set: a
plain Markdown string is what RunResult.final_output already is.
"""

import json

from agents import Agent, Runner, trace

from .config import MODEL
from .statement_ordering import STATEMENT_SUBMODULES

# Kept out of the f-string below: the LaTeX examples contain literal
# braces ( \frac{a}{b}, V_{1} ) that an f-string parser would read as
# replacement fields.
ANSWER_FORMATTING_RULES = """\
Formatting (reply in normal Markdown, clean and skimmable):

- Use "- " bullet points for any list of figures, drivers, or comparisons -
  one point per line. Do not use a bullet for a single item.
- Add a "## " or "### " heading only when the answer has two or more
  distinct sections. Skip headings for a short answer.
- Use a Markdown table when comparing the same metric across several periods.
- Put key numbers in **bold** (e.g. **$4.2 billion**). Never bold a whole
  sentence.
- Write every mathematical expression in LaTeX: \\( ... \\) inline and
  \\[ ... \\] for a displayed equation. For example a growth rate is
  \\( \\frac{V_{1} - V_{0}}{V_{0}} \\times 100\\% \\).
- Do NOT use $ ... $ or $$ ... $$ for math: a bare $ means US dollars here.
  Write currency as plain text, e.g. $5.2 billion.
- Do not add filler such as "I hope this helps" and do not restate the
  question.
- If you see a number like 1,000,000,000, then convert it to 1 billion and write it in words, write 1,000,000 as 1 million and write 1,000 as 1 grand.
"""

_ANALYST_INSTRUCTIONS = f"""
You are a financial research assistant.

Answer the user's question using ONLY the Yahoo Finance
data supplied in the input below.

Instructions:.
1. Answer the actual question directly.
2. Do not claim information that is not present in the Yahoo Finance data.
3. {STATEMENT_SUBMODULES} are financial statements. Always display them in a table comparing the same metric across several periods.

5. If the data contains several years, compare them.
6. If a value is missing or null, say that the Yahoo Finance data
   does not provide it.
8. Include as much information as poosible fron yahoo finance data.
9. Make sure you write text more than 200 words on the answer.
7. Do not fabricate numbers.
10. If "valuation_metrics" is present, use those pre-computed values as-is —
   do not recalculate them from other fields. For every ratio you cite,
   show its "formula" with the numbers from "inputs" substituted and then
   the result, e.g. \\( \\text{{P/E}} = \\frac{{2500}}{{85}} = 29.4 \\). If a
   metric's "value" is null, say it cannot be computed and give the reason
   from its "note". Lead with the ratios whose "applicable_for_sector" is
   true; for the rest, note they are less meaningful for this company's
   sector.
{ANSWER_FORMATTING_RULES}
Give a concise but useful financial answer.
"""

_answer_agent = Agent(
    name="Equity Research Analyst",
    instructions=_ANALYST_INSTRUCTIONS,
    model=MODEL,
)


def generate_answer(question: str, intent: dict, yahoo_data) -> str:

    data_json = json.dumps(yahoo_data, indent=2, ensure_ascii=False)

    dynamic_input = f"""
User question:
{question}

Detected intent:
{json.dumps(intent, indent=2)}

Yahoo Finance data:
{data_json}
"""

    with trace("generate_answer", metadata={"question": question[:200]}):
        result = Runner.run_sync(_answer_agent, dynamic_input)
    return result.final_output
