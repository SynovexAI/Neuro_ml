"use client";

import { useState } from "react";
import RagLab from "@/components/RagLab";
import RagFlowLab from "@/components/RagFlowLab";
import RagPlayground from "@/components/RagPlayground";

// Wraps the RAG lab with a Steps | Canvas toggle — the classic guided stepper
// stays; Canvas is the new node-pipeline edition on the same real engine.
export default function RagLabTabs() {
  const [tab, setTab] = useState<"steps" | "canvas" | "practice">("steps");
  return (
    <>
      <div className="seg" style={{ maxWidth: 360, marginBottom: 14 }}>
        <button className={tab === "steps" ? "on" : ""} onClick={() => setTab("steps")}>Steps</button>
        <button className={tab === "canvas" ? "on" : ""} onClick={() => setTab("canvas")}>Canvas</button>
        <button className={tab === "practice" ? "on" : ""} onClick={() => setTab("practice")}>Practice Playground</button>
      </div>
      {tab === "steps" && <RagLab />}
      {tab === "canvas" && <RagFlowLab />}
      {tab === "practice" && <RagPlayground />}
    </>
  );
}
