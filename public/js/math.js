// Typeset any \( ... \) / \[ ... \] math the backend left in the answer.
// KaTeX only; we intentionally do NOT register $...$ so currency like
// "$5.2 billion" is never mistaken for math. Purely visual — a failure
// here must never blank out the answer text.
function typesetMath(el) {
  if (typeof renderMathInElement !== "function") return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  } catch (err) {
    /* leave the raw \( ... \) visible rather than breaking the card */
  }
}
