"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "@/lib/toast";

type Tpl = { id: string; lab: string; href: string; icon: string; title: string; tag: string; desc: string; config: Record<string, unknown> };

const POLICY = "Returns policy: damaged items may be returned within 30 days of delivery for a full refund. Refunds go to the original payment method within 5 business days. Shipping is free on orders over $50, otherwise a $6 flat fee applies. Gift cards are non-refundable. Refunds above $500 require a manager's approval.";

const TEMPLATES: Tpl[] = [
  // Agent
  { id: "agent-support", lab: "agent", href: "/labs/agent", icon: "🎧", title: "Support agent", tag: "agent · ReAct",
    desc: "Answers refund/shipping questions from a knowledge base and asks a human before big refunds.",
    config: { type: "react", name: "Support agent", description: "Grounded support with human approval.", systemPrompt: "You are a friendly customer-support agent. Answer refund & shipping questions grounded in the knowledge base, and use human_approval before authorising any refund over $500.", provider: "", model: "", temperature: 0.3, maxIterations: 6, tools: ["knowledge", "human_approval"], knowledge: POLICY, task: "A customer says their $650 order arrived damaged and wants a full refund. What should we do?" } },
  { id: "agent-research", lab: "agent", href: "/labs/agent", icon: "🔎", title: "Research assistant", tag: "agent · tools",
    desc: "Fetches a web page and returns a concise, cited summary.",
    config: { type: "react", name: "Research assistant", description: "Reads the web and summarises.", systemPrompt: "You research questions using web_fetch and reply with a concise, well-structured answer. Cite the page you used.", provider: "", model: "", temperature: 0.4, maxIterations: 6, tools: ["web_fetch", "calculator"], task: "Fetch https://en.wikipedia.org/wiki/Retrieval-augmented_generation and summarise RAG in 3 bullet points." } },
  // RAG
  { id: "rag-policy", lab: "rag", href: "/labs/rag", icon: "❖", title: "Policy Q&A bot", tag: "rag",
    desc: "Chunk a policy doc, embed it, and answer questions grounded in it.",
    config: { docs: [{ name: "returns-policy.txt", kind: "txt", text: POLICY + " Store hours are 9am–6pm on weekdays. International orders take 10–15 business days. Warranty claims for electronics are handled by the manufacturer for the first 12 months." }], size: 60, overlap: 12, strategy: "hybrid", topK: 3, question: "What is the refund window for damaged items?" } },
  // Prompting
  { id: "prompt-summary", lab: "prompting", href: "/labs/prompting", icon: "✎", title: "3-bullet summarizer", tag: "prompt",
    desc: "A tight system prompt that always returns exactly three short bullets.",
    config: { system: "You are a precise summarizer. Output exactly 3 bullet points, each under 15 words. No preamble.", prompt: "Summarize this: Retrieval-augmented generation retrieves relevant documents and feeds them to an LLM so answers are grounded in your data instead of the model's memory.", temp: 0.3, maxTokens: 300, topP: 1, freqPenalty: 0, presencePenalty: 0, model: "" } },
  { id: "prompt-json", lab: "prompting", href: "/labs/prompting", icon: "{ }", title: "JSON extractor", tag: "prompt · structured",
    desc: "Extract fields as strict JSON — a starting point for structured output.",
    config: { system: "Extract fields as strict minified JSON with keys {name, email, intent}. Output only JSON, no prose.", prompt: "Hi, I'm Dana (dana@acme.io) — I'd like to cancel my subscription.", temp: 0, maxTokens: 200, topP: 1, freqPenalty: 0, presencePenalty: 0, model: "" } },
  // ML
  { id: "ml-churn", lab: "ml", href: "/labs/ml", icon: "⚙", title: "Churn classifier", tag: "ml · Random Forest",
    desc: "Predict customer churn on the sample dataset with a Random Forest.",
    config: { dsName: "churn", task: "classification", algo: "RandomForest" } },
  // DL
  { id: "dl-spiral", lab: "dl", href: "/labs/dl", icon: "⟐", title: "Spiral classifier", tag: "dl · MLP",
    desc: "A 2-layer network that learns the classic two-spiral boundary.",
    config: { kind: "spiral", noise: 0.1, hidden: [8, 8], act: "tanh", lr: 0.03, l2: 0 } },
  { id: "dl-xor", lab: "dl", href: "/labs/dl", icon: "⟐", title: "XOR network", tag: "dl · MLP",
    desc: "The smallest net that solves a non-linear XOR — great for intuition.",
    config: { kind: "xor", noise: 0, hidden: [4], act: "tanh", lr: 0.05, l2: 0 } },
  // ETL / ELT
  { id: "etl-orders", lab: "etl", href: "/labs/etl", icon: "⇉", title: "Clean & aggregate orders", tag: "etl",
    desc: "Filter to paid orders, then sum revenue by region — a classic ETL pipeline.",
    config: { srcName: "orders (e-commerce)", pipeMode: "etl", mode: "batch", rules: [], models: [], ops: [{ id: "t1", type: "filter", col: "status", op: "==", value: "paid" }, { id: "t2", type: "aggregate", groupBy: "region", agg: "sum", aggCol: "amount" }] } },
  { id: "elt-revenue", lab: "etl", href: "/labs/etl", icon: "⇉", title: "ELT: revenue by region", tag: "elt · SQL",
    desc: "Land raw orders, then transform in SQL (dbt-style) to revenue per region.",
    config: { srcName: "orders (e-commerce)", pipeMode: "elt", mode: "batch", rules: [], models: [], ops: [] } },
];

const GROUPS: { lab: string; label: string }[] = [
  { lab: "agent", label: "Agents" },
  { lab: "rag", label: "RAG" },
  { lab: "prompting", label: "Prompting" },
  { lab: "ml", label: "Machine learning" },
  { lab: "dl", label: "Deep learning" },
  { lab: "etl", label: "ETL / ELT" },
];

export default function TemplatesGallery() {
  const router = useRouter();
  const [busy, setBusy] = useState("");

  async function use(t: Tpl) {
    setBusy(t.id);
    try {
      const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: t.lab, name: t.title, config: t.config }) });
      const j = await r.json().catch(() => null);
      if (r.ok && j?.id) router.push(`${t.href}?project=${j.id}`);
      else { toast(j?.error || "Could not create the project", "error"); setBusy(""); }
    } catch (e) { toast((e as Error).message, "error"); setBusy(""); }
  }

  return (
    <>
      <div className="teach-note"><span className="ic">✦</span><span>Each template creates a ready-to-run project and opens it in the right Lab — edit anything, then run, save, or publish.</span></div>
      {GROUPS.map((g) => {
        const items = TEMPLATES.filter((t) => t.lab === g.lab);
        if (!items.length) return null;
        return (
          <div key={g.lab} style={{ marginBottom: 22 }}>
            <div className="sec-title">{g.label}</div>
            <div className="card-grid">
              {items.map((t) => (
                <div key={t.id} className="agent-card" style={{ cursor: "default" }}>
                  <div className="agent-card-top"><span className="agent-ic">{t.icon}</span><b>{t.title}</b></div>
                  <div className="note" style={{ marginTop: 4 }}>{t.tag}</div>
                  <p className="page-sub" style={{ fontSize: 12.5, margin: "8px 0 12px", lineHeight: 1.5 }}>{t.desc}</p>
                  <button className="btn sm" style={{ marginTop: "auto", alignSelf: "flex-start" }} onClick={() => use(t)} disabled={!!busy}>{busy === t.id ? "Opening…" : "Use template →"}</button>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </>
  );
}
