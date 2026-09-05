"""
Generate the final Markdown answer from a question, its detected intent,
and the fetched Yahoo Finance data.
"""

import json
from datetime import date

from .config import MODEL, client
from .statement_ordering import STATEMENT_SUBMODULES

# Kept out of the f-string prompt below: the LaTeX example contains braces
# ( \frac{a}{b}, V_{1} ) that Python's f-string parser would read as fields.
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


def generate_answer(question: str, intent: dict, yahoo_data) -> str:

    data_json = json.dumps(yahoo_data, indent=2, ensure_ascii=False)

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the Yahoo Finance
data supplied below.

User question:
{question}

Detected intent:
{json.dumps(intent, indent=2)}

Yahoo Finance data:
{data_json}

Instructions:.
1. Answer the actual question directly.
2. Do not claim information that is not present in the Yahoo Finance data.
3. {STATEMENT_SUBMODULES} are financial statements. Always display them in a table comparing the same metric across several periods.

4.  Always comapre Operating Cash Flow, Free Cash Flow, Cash Flow ,
   Investing, Cash Flow from Financing, and  Change in Cash in a tabular format.

5. If the data contains several years, compare them.
6. If a value is missing or null, say that the Yahoo Finance data
   does not provide it.
8. Include as much information as poosible fron yahoo finance data.
9. Make sure you write text more than 200 words on the answer.
7. Do not fabricate numbers.
{ANSWER_FORMATTING_RULES}
Give a concise but useful financial answer.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text
