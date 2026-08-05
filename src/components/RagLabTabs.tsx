"use client";

import { useState } from "react";
import RagLab from "@/components/RagLab";
import RagFlowLab from "@/components/RagFlowLab";

// Wraps the RAG lab with a Steps | Canvas toggle — the classic guided stepper
// stays; Canvas is the new node-pipeline edition on the same real engine.
export default function RagLabTabs() {
  const [tab, setTab] = useState<"steps" | "canvas">("steps");
  return (
    <>
      <div className="seg" style={{ maxWidth: 240, marginBottom: 14 }}>
        <button className={tab === "steps" ? "on" : ""} onClick={() => setTab("steps")}>Steps</button>
        <button className={tab === "canvas" ? "on" : ""} onClick={() => setTab("canvas")}>Canvas</button>
      </div>
      {tab === "steps" ? <RagLab /> : <RagFlowLab />}
    </>
  );
}
