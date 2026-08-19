"use client";

import katex from "katex";
import "katex/dist/katex.min.css";

// Renders a LaTeX string with KaTeX. `block` = display mode (centered, larger).
// KaTeX output is trusted HTML (we pass throwOnError:false so bad input degrades).
export default function Katex({ tex, block = false }: { tex: string; block?: boolean }) {
  let html = "";
  try { html = katex.renderToString(tex, { throwOnError: false, displayMode: block, output: "html" }); }
  catch { html = tex; }
  return <span className={block ? "kx-block" : "kx"} dangerouslySetInnerHTML={{ __html: html }} />;
}
