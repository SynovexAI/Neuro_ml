// Multimodal RAG — one index across text, tables, and images.
//
// The idea: a question's answer often lives in a chart or a table row, not in
// prose. Plain RAG indexes text only, so those answers are unreachable. Multimodal
// RAG puts every modality into ONE shared index so a single query can rank a
// table row above a paragraph.
//
// Honest scope note: a production system embeds pixels with a vision encoder
// (CLIP, ColPali) so it can match a query against a chart it has never had
// described. This module builds the *unified index* half for real — tables are
// serialised to retrievable row text, images carry their caption/OCR/alt text —
// but it has no vision encoder, so an image is only findable through the text
// attached to it. The UI says so rather than implying pixel search.

import { type RagIndex, buildIndex, retrieve, type Strategy } from "./ragUtils";

export type Modality = "text" | "table" | "image";

export interface ModalItem {
  modality: Modality;
  /** Human label shown in results — a filename, a table caption, a heading. */
  title: string;
  /** The retrievable text for this item. */
  content: string;
  /** For tables: the parsed rows, so a hit can be shown as a table not a blob. */
  rows?: string[][];
  /** For images: where the searchable text came from. */
  textSource?: "caption" | "ocr" | "alt";
}

export interface ModalIndex {
  idx: RagIndex;
  items: ModalItem[];
}

/**
 * Serialise a table into one retrievable string per row, each carrying its
 * column headers. "Revenue | 2024 | $2.42M" is far more retrievable than a
 * bare "2.42", because the header words are what a question actually contains.
 */
export function tableToRowTexts(rows: string[][], caption?: string): string[] {
  if (rows.length === 0) return [];
  const [header, ...body] = rows;
  if (body.length === 0) return [[caption, header.join(" ")].filter(Boolean).join(" — ")];
  return body.map((r) =>
    [caption, ...r.map((cell, c) => `${header[c] ?? `col${c + 1}`}: ${cell}`)].filter(Boolean).join(" · "),
  );
}

/** Parse a pasted CSV/TSV/markdown-pipe block into rows. */
export function parseTable(raw: string): string[][] {
  const lines = raw.trim().split(/\r?\n/).filter((l) => l.trim() && !/^\s*\|?\s*:?-{2,}/.test(l));
  const delim = lines[0]?.includes("|") ? "|" : lines[0]?.includes("\t") ? "\t" : ",";
  return lines.map((l) =>
    l.split(delim).map((c) => c.trim()).filter((c, i, a) => !(c === "" && (i === 0 || i === a.length - 1))),
  ).filter((r) => r.length > 0);
}

/**
 * Build one index over every modality. Each item contributes exactly one
 * document, so ranks are directly comparable across modalities — that shared
 * ranking is the whole point of a unified index.
 */
export function buildMultimodalIndex(items: ModalItem[]): ModalIndex {
  return { idx: buildIndex(items.map((it) => `${it.title} ${it.content}`)), items };
}

export interface ModalHit {
  i: number;
  score: number;
  item: ModalItem;
  modality: Modality;
}

/** Retrieve across all modalities at once, optionally filtered to a subset. */
export function retrieveMultimodal(
  mi: ModalIndex,
  query: string,
  k: number,
  opts: { strategy?: Strategy; only?: Modality[] } = {},
): { hits: ModalHit[]; perModality: Record<Modality, number> } {
  const { strategy = "hybrid", only } = opts;
  const wide = retrieve(mi.idx, query, strategy, mi.items.length);
  const filtered = wide.filter((r) => {
    const m = mi.items[r.i]?.modality;
    return !!m && (!only || only.includes(m));
  });
  const hits: ModalHit[] = filtered.slice(0, k).map((r) => ({ i: r.i, score: r.score, item: mi.items[r.i], modality: mi.items[r.i].modality }));
  const perModality: Record<Modality, number> = { text: 0, table: 0, image: 0 };
  hits.forEach((h) => { perModality[h.modality] += 1; });
  return { hits, perModality };
}

/** A small worked corpus that has a genuine answer in each modality. */
export function sampleMultimodalCorpus(): ModalItem[] {
  const rows = [
    ["Quarter", "Revenue", "Margin", "Region"],
    ["Q1 2024", "$1.82M", "31%", "EMEA"],
    ["Q2 2024", "$2.05M", "34%", "EMEA"],
    ["Q3 2024", "$2.31M", "36%", "APAC"],
    ["Q4 2024", "$2.42M", "38%", "APAC"],
  ];
  return [
    {
      modality: "text",
      title: "Annual report — outlook",
      content:
        "The company expects continued expansion through the next financial year, driven primarily by subscription renewals and a broadening partner network across the region.",
    },
    {
      modality: "text",
      title: "Annual report — risk factors",
      content:
        "Principal risks include supply chain concentration, currency exposure in emerging markets, and dependence on a small number of enterprise customers.",
    },
    ...tableToRowTexts(rows, "Quarterly results 2024").map<ModalItem>((content, r) => ({
      modality: "table",
      title: `Quarterly results 2024 — row ${r + 1}`,
      content,
      rows: [rows[0], rows[r + 1]],
    })),
    {
      modality: "image",
      title: "figure-3-margin-trend.png",
      content:
        "Line chart titled Margin trend by quarter. The vertical axis shows gross margin percentage from 30 to 40 percent. The line rises steadily from 31 percent in Q1 to 38 percent in Q4 2024.",
      textSource: "caption",
    },
    {
      modality: "image",
      title: "figure-7-regional-split.png",
      content:
        "Pie chart of revenue by region for 2024. APAC accounts for the largest share at 54 percent, EMEA 33 percent, and the Americas 13 percent.",
      textSource: "ocr",
    },
  ];
}
