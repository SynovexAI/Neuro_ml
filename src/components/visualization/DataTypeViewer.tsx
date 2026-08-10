import React from "react";
import { Table } from "@/lib/etlUtils";

interface DataTypeViewerProps {
  table: Table;
}

export default function DataTypeViewer({ table }: DataTypeViewerProps) {
  const inferType = (col: string) => {
    const values = table.rows.map((r) => r[col]).filter((v) => v !== null && v !== undefined);
    if (values.length === 0) return "unknown";

    let isNum = true;
    let isBool = true;
    let isDate = true;

    for (const val of values) {
      const str = String(val);
      if (isNaN(Number(val))) {
        isNum = false;
      }
      if (str !== "true" && str !== "false" && typeof val !== "boolean") {
        isBool = false;
      }
      if (isNaN(Date.parse(str))) {
        isDate = false;
      }
    }

    if (isNum) return "number";
    if (isBool) return "boolean";
    if (isDate) return "date";
    return "string";
  };

  const getBadgeStyle = (type: string) => {
    switch (type) {
      case "number":
        return { bg: "rgba(59, 130, 246, 0.15)", color: "#60a5fa", label: "🔢 NUMBER" };
      case "boolean":
        return { bg: "rgba(16, 185, 129, 0.15)", color: "#34d399", label: "✅ BOOLEAN" };
      case "date":
        return { bg: "rgba(245, 158, 11, 0.15)", color: "#fbbf24", label: "📅 DATE" };
      case "string":
        return { bg: "rgba(168, 85, 247, 0.15)", color: "#c084fc", label: "🔤 STRING" };
      default:
        return { bg: "rgba(255, 255, 255, 0.05)", color: "var(--faint)", label: "❓ UNKNOWN" };
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 14 }}>
      {table.cols.map((col) => {
        const type = inferType(col);
        const style = getBadgeStyle(type);
        const samples = table.rows
          .slice(0, 3)
          .map((r) => (r[col] == null ? "—" : String(r[col])));

        return (
          <div key={col} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: 8, padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontWeight: "bold", fontSize: 14, color: "#f3f4f6" }}>{col}</div>
            
            <div style={{ display: "inline-block", alignSelf: "flex-start", padding: "4px 8px", borderRadius: 6, fontSize: 11, fontWeight: "bold", background: style.bg, color: style.color }}>
              {style.label}
            </div>

            <div style={{ marginTop: 4 }}>
              <span style={{ fontSize: 11, color: "var(--faint)", display: "block", marginBottom: 2 }}>Sample Values:</span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {samples.map((s, idx) => (
                  <span key={idx} style={{ background: "rgba(255,255,255,0.05)", border: "1px solid var(--border)", padding: "2px 6px", borderRadius: 4, fontSize: 11, color: "#d1d5db" }}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
