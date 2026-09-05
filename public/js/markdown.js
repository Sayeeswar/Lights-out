// Minimal, safe Markdown -> HTML for the FALLBACK path only (when the backend
// didn't send pre-rendered answer_html). Every line is HTML-escaped first, and
// the only tags produced are the fixed set this function writes itself, so no
// markup from `src` can slip through. Handles: # headings, - / * / 1. lists,
// blank-line paragraphs, single newline -> <br>, **bold**, *italic*, `code`.
// Tables are NOT handled here — that needs the backend renderer.
function renderMarkdown(src) {
  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

  const lines = String(src).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let list = null; // "ul" | "ol" | null
  let para = [];

  const flushPara = () => {
    if (para.length) {
      out.push("<p>" + para.map(inline).join("<br>") + "</p>");
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      out.push("</" + list + ">");
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");

    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      const level = heading[1].length;
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushPara();
      const want = bullet ? "ul" : "ol";
      if (list !== want) {
        flushList();
        list = want;
        out.push("<" + want + ">");
      }
      out.push("<li>" + inline((bullet || numbered)[1]) + "</li>");
      continue;
    }

    flushList();
    para.push(line);
  }

  flushPara();
  flushList();
  return out.join("\n");
}
