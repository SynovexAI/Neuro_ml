import React from "react";
import { Table } from "@/lib/etlUtils";

interface MissingValueViewerProps {
  table: Table;
}

export default function MissingValueViewer({ table }: MissingValueViewerProps) {
  const getMissingData = () => {
    const total = table.rows.length;
    return table.cols.map((col) => {
      const missing = table.rows.filter(
        (r) => r[col] === null || r[col] === undefined || String(r[col]).trim() === ""
      ).length;
      const filled = total - missing;
      const missingPct = total > 0 ? (missing / total) * 100 : 0;
      return { col, missing, filled, missingPct };
    });
  };

  const missingList = getMissingData();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {missingList.map((m) => {
        const filledPct = 100 - m.missingPct;
        const barColor = m.missingPct > 20 ? "#ef4444" : m.missingPct > 0 ? "#f59e0b" : "#10b981";
        
        return (
          <div key={m.col} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: 8, padding: 14 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div>
                <strong style={{ fontSize: 14, color: "#f3f4f6" }}>{m.col}</strong>
                <span style={{ fontSize: 12, color: "var(--faint)", marginLeft: 10 }}>
                  ({m.filled} filled · {m.missing} missing)
                </span>
              </div>
              <span style={{ fontSize: 13, fontWeight: "bold", color: barColor }}>
                {m.missingPct.toFixed(1)}% missing
              </span>
            </div>
            
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: 6, height: 12, overflow: "hidden" }}>
                <div style={{ background: barColor, width: `${filledPct}%`, height: "100%", transition: "width 0.3s ease" }} />
              </div>
              <span style={{ fontSize: 12, width: 40, textAlign: "right", color: "var(--faint)" }}>
                {filledPct.toFixed(0)}% OK
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
