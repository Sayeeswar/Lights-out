# Module spec: `answer.py`

**File this describes:** `lib/yahoo_finance/answer.py`
**Depended on by:** `pipeline.py`, which calls
`generate_answer(question, intent, yahoo_data)` and passes the returned
string straight into `markdown_to_safe_html(answer)` and into the API
response's `"answer"` field.

## What this file does today

One function, `generate_answer(question, intent, yahoo_data) -> str`:

1. Serializes `yahoo_data` to a JSON string.
2. Builds a large prompt string combining: the role framing, the question,
   the detected intent (as JSON), the fetched data (as JSON), a numbered
   list of answer-writing instructions, and `ANSWER_FORMATTING_RULES` (a
   separate constant covering Markdown/LaTeX formatting rules, kept outside
   the f-string because it contains literal `{` `}` characters from LaTeX
   that would otherwise be misread as f-string placeholders).
3. Calls `client.responses.create(model=MODEL, input=prompt)`.
4. Returns `response.output_text` directly — no parsing at all, because
   this one is meant to be free-form Markdown text, not JSON.

## What changes, and why

This is the simpler of the two conversions, precisely *because* this
function doesn't parse anything today. There's no `output_type` to design —
you want a plain string back, and that's exactly what
`RunResult.final_output` is when the `Agent` doesn't set `output_type`.

The split is the same idea as `intent.py`: the parts of the prompt that
don't depend on the specific question — the role framing, the numbered
instructions, `ANSWER_FORMATTING_RULES` — become the `Agent`'s
`instructions`, built once at import time. The parts that change every
call — the question, the intent, the fetched data — become the `input`
string passed to `Runner.run_sync(...)`.

One small wrinkle: today's numbered instruction list includes a line that
references `STATEMENT_SUBMODULES` (a set imported from
`statement_ordering.py`) directly inside the f-string:

```python
3. {STATEMENT_SUBMODULES} are financial statements. Always display them in a table comparing the same metric across several periods.
```

`STATEMENT_SUBMODULES` doesn't change per-question — it's a fixed set
defined at module load time — so this line can also move into the
one-time `instructions` string. This is captured correctly below.

## Full new file content

```python
"""
Generate the final Markdown answer from a question, its detected intent,
and the fetched Yahoo Finance data.
"""

import json

from agents import Agent, Runner

from .config import MODEL
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

_ANALYST_INSTRUCTIONS = f"""
You are a financial research assistant.

Answer the user's question using ONLY the Yahoo Finance
data supplied in the input below.

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

    result = Runner.run_sync(_answer_agent, dynamic_input)
    return result.final_output
```

**A precise detail:** the doubled braces `{{"formula"}}`-style escaping in
rule 10's LaTeX example (`\\text{{P/E}}`, `\\frac{{2500}}{{85}}`) is *new*
compared to the original file. The original file avoided this problem by
keeping `ANSWER_FORMATTING_RULES` as a separate, non-f-string constant —
but rule 10, which also contains literal LaTeX braces, was already living
*inside* the f-string in the original `answer.py`. Check the real,
current `answer.py` for the exact current escaping before copying this
block — if rule 10's braces aren't already doubled in the live file, this
spec's version fixes a latent bug (a literal `{` in an f-string that isn't
doubled raises `KeyError` or is misread as a field reference the moment the
module is imported). If the live file runs today without error, that
means it's already using some form of escaping there — match whatever
that file actually does, character for character, rather than trusting
this transcription blindly.

## What must NOT change

- The exact wording of every numbered instruction and every line in
  `ANSWER_FORMATTING_RULES` — these encode real product requirements (word
  count minimum, table formatting, LaTeX-not-dollar-signs, the unit-word
  conversion rule).
- The fact that `generate_answer()` returns a plain `str`.
- The order of information in the `input` string (question, then intent,
  then data) — while an LLM is less order-sensitive than code, keeping it
  identical removes one variable when comparing before/after answers.

## How to verify

Run the exact same question through both the old and new `generate_answer`
(see `04-testing-and-rollout.md`) and compare: does the answer still
contain a table for statement data, LaTeX-formatted math, bolded numbers,
and stay above ~200 words? You're not looking for byte-identical output —
the model can phrase things differently — you're looking for the same
*shape and rules* being followed.
