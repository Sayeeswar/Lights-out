"""
Turn the model's raw Markdown answer into sanitized, display-ready HTML,
and normalize inline math so the frontend's KaTeX auto-render can typeset it.

Why this lives in the backend
-----------------------------
CLAUDE.md requires every bit of functionality to live in the backend; the
frontend only paints pixels. So Markdown -> HTML conversion and HTML
sanitization happen here, and the browser merely injects the result and runs
KaTeX (a purely visual typesetting step, the same way Chart.js turns backend
chart data into a drawing).

Pipeline (see ``markdown_to_safe_html``)
    1. normalize_math_delimiters : fold the various ways the model writes math
                                   into one canonical delimiter pair.
    2. _protect_math             : lift math spans out so the Markdown parser
                                   cannot mangle the ``*  _  \\`` inside them.
    3. markdown.markdown         : Markdown -> HTML.
    4. _restore_math             : splice the untouched math spans back in.
    5. bleach.clean             : drop anything not on the tag/attr allowlist.
"""

import re

import bleach
import markdown as md

# ---------------------------------------------------------------------------
# Canonical math delimiters
# ---------------------------------------------------------------------------
# We deliberately DO NOT treat ``$...$`` as math. This is a finance assistant,
# so "$5.2 billion" appears constantly and must render as literal currency,
# never as a broken math expression. Everything is folded to these two pairs,
# and the frontend's KaTeX config accepts only these two.
INLINE_MATH = (r"\(", r"\)")
BLOCK_MATH = (r"\[", r"\]")

# ---------------------------------------------------------------------------
# Markdown configuration
# ---------------------------------------------------------------------------
MARKDOWN_EXTENSIONS = [
    "extra",       # tables, fenced code, footnotes, attr_list, def lists, ...
    "sane_lists",  # don't let a "1." start an <ol> in the middle of a <ul>
    "nl2br",       # a single newline -> <br>; chat answers rely on this
]

# ---------------------------------------------------------------------------
# Sanitization allowlist -- anything not named here is stripped by bleach.
# No <a>, <img>, <script>, <style>, no event handlers, no inline styles.
# ---------------------------------------------------------------------------
ALLOWED_TAGS = [
    "p", "br", "hr",
    "strong", "b", "em", "i", "u", "s", "del", "mark", "sub", "sup",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "blockquote", "code", "pre",
    "table", "thead", "tbody", "tr", "th", "td",
    "span", "div",
]

ALLOWED_ATTRIBUTES = {
    "*": ["class"],
    "th": ["align"],
    "td": ["align"],
}

# A token the Markdown parser will pass through untouched (no metacharacters)
# and that will never collide with real answer text.
_MATH_PLACEHOLDER = "MATHPLACEHOLDER{index}ENDMATHPLACEHOLDER"

# A LaTeX environment such as \begin{aligned} ... \end{aligned}. These are
# valid display math on their own, but KaTeX auto-render only looks *inside*
# \( \) / \[ \], so an unwrapped one would show as raw source.
_LATEX_ENV = re.compile(r"\\begin\{([a-zA-Z*]+)\}.*?\\end\{\1\}", re.DOTALL)


def normalize_math_delimiters(text: str) -> str:
    """
    Fold the many ways the model might delimit math into the two canonical
    pairs declared above (``INLINE_MATH`` and ``BLOCK_MATH``), so that
    ``_protect_math`` and the frontend's KaTeX config only ever deal with one
    form.

    Mapping:
      * ``$$ ... $$``                       -> block math   ``\\[ ... \\]``
      * ``\\begin{env} ... \\end{env}`` (when not already wrapped)
                                             -> block math   ``\\[ ... \\]``
      * ``\\( ... \\)`` / ``\\[ ... \\]``    -> already canonical, untouched
      * a single ``$ ... $``                 -> LEFT ALONE. In a finance bot a
        lone ``$`` is almost always currency ("$5B"), and treating it as math
        breaks far more often than it helps.
    """
    if not text:
        return text

    # 1. $$ ... $$  ->  \[ ... \]
    #    `\$\$` needs two literal dollars, so a lone `$5` can never match.
    #    Non-greedy + DOTALL so one equation (possibly multi-line) is taken
    #    at a time and a stray unpaired `$$` is left as-is.
    text = re.sub(r"\$\$(.+?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)

    # 2. Wrap a bare \begin{env} ... \end{env}, unless it already sits inside
    #    \[ \] or \( \) (e.g. it arrived that way, or step 1 just wrapped it).
    def _wrap(match: "re.Match") -> str:
        body = match.group(0)
        before = text[:match.start()].rstrip()
        after = text[match.end():].lstrip()
        if before.endswith(("\\[", "\\(")) or after.startswith(("\\]", "\\)")):
            return body
        return "\\[" + body + "\\]"

    text = _LATEX_ENV.sub(_wrap, text)

    return text


def _protect_math(text: str):
    """
    Replace every canonical math span with a placeholder token before the
    Markdown parser runs, so it can't touch the ``*``, ``_`` and ``\\`` inside.

    Returns ``(text_with_placeholders, [original_spans])``.
    """
    spans: list[str] = []

    def stash(match: "re.Match") -> str:
        spans.append(match.group(0))
        return _MATH_PLACEHOLDER.format(index=len(spans) - 1)

    inline_l, inline_r = (re.escape(d) for d in INLINE_MATH)
    block_l, block_r = (re.escape(d) for d in BLOCK_MATH)

    # Block first: \[ ... \] can contain sequences that look like \( ... \).
    text = re.sub(rf"{block_l}.+?{block_r}", stash, text, flags=re.DOTALL)
    text = re.sub(rf"{inline_l}.+?{inline_r}", stash, text, flags=re.DOTALL)
    return text, spans


def _restore_math(html: str, spans: list) -> str:
    """Splice the original math spans back in after Markdown conversion."""
    for index, span in enumerate(spans):
        html = html.replace(_MATH_PLACEHOLDER.format(index=index), span)
    return html


def markdown_to_safe_html(text: str) -> str:
    """
    Convert a raw Markdown answer string into sanitized HTML that the chat UI
    can inject directly. Math spans survive untouched for KaTeX to typeset in
    the browser. Returns "" for empty input.
    """
    if not text or not text.strip():
        return ""

    # bleach keeps the *text* inside a stripped <script>/<style>; drop the
    # whole block up front so it never shows up as stray page content.
    text = re.sub(
        r"<\s*(script|style)\b.*?<\s*/\s*\1\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = normalize_math_delimiters(text)
    text, math_spans = _protect_math(text)

    html = md.markdown(text, extensions=MARKDOWN_EXTENSIONS, output_format="html5")
    html = _restore_math(html, math_spans)

    return bleach.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True,
    )
