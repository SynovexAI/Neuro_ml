import React from "react";
import { Table } from "@/lib/etlUtils";

interface StatisticsPanelProps {
  table: Table;
}

export default function StatisticsPanel({ table }: StatisticsPanelProps) {
  const getStats = () => {
    const stats: Array<{
      col: string;
      count: number;
      nulls: number;
      distinct: number;
      min: string;
      max: string;
      mean: string;
      isNumeric: boolean;
      nullPct: number;
    }> = [];

    table.cols.forEach((col) => {
      const values = table.rows.map((r) => r[col]);
      const nonNulls = values.filter((v) => v !== null && v !== undefined);
      const distinct = new Set(nonNulls).size;
      
      const numValues = nonNulls
        .map((v) => Number(v))
        .filter((v) => !isNaN(v));

      const isNumeric = numValues.length > 0 && numValues.length === nonNulls.length;

      let min = "—";
      let max = "—";
      let mean = "—";

      if (numValues.length > 0) {
        min = Math.min(...numValues).toFixed(1);
        max = Math.max(...numValues).toFixed(1);
        mean = (numValues.reduce((a, b) => a + b, 0) / numValues.length).toFixed(1);
      } else if (nonNulls.length > 0) {
        const sorted = [...nonNulls].map(String).sort();
        min = sorted[0].slice(0, 10);
        max = sorted[sorted.length - 1].slice(0, 10);
      }

      const nullPct = values.length > 0 ? ((values.length - nonNulls.length) / values.length) * 100 : 0;

      stats.push({
        col,
        count: values.length,
        nulls: values.length - nonNulls.length,
        distinct,
        min,
        max,
        mean,
        isNumeric,
        nullPct,
      });
    });

    return stats;
  };

  const statsList = getStats();

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 16 }}>
      {statsList.map((s) => (
        <div key={s.col} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: 8, padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontSize: 14, fontWeight: "bold", color: "#f3f4f6" }}>{s.col}</span>
            <span className="badge" style={{ fontSize: 10, background: s.isNumeric ? "rgba(59, 130, 246, 0.15)" : "rgba(168, 85, 247, 0.15)", color: s.isNumeric ? "#60a5fa" : "#c084fc", border: "1px solid transparent" }}>
              {s.isNumeric ? "NUMERIC" : "CATEGORICAL"}
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
            {/* Visual indicator for data presence */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", color: "var(--faint)", marginBottom: 4 }}>
                <span>Completion</span>
                <span>{(100 - s.nullPct).toFixed(0)}%</span>
              </div>
              <div style={{ background: "var(--border)", height: 6, borderRadius: 3, overflow: "hidden" }}>
                <div style={{ background: s.nullPct > 20 ? "var(--crit, #ef4444)" : "var(--accent, #3dec7f)", width: `${100 - s.nullPct}%`, height: "100%" }} />
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
              <div>
                <span style={{ color: "var(--faint)", display: "block" }}>Distinct Values</span>
                <strong style={{ fontSize: 13 }}>{s.distinct}</strong>
              </div>
              {s.isNumeric && (
                <div>
                  <span style={{ color: "var(--faint)", display: "block" }}>Mean (Avg)</span>
                  <strong style={{ fontSize: 13, color: "#60a5fa" }}>{s.mean}</strong>
                </div>
              )}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 4 }}>
              <div>
                <span style={{ color: "var(--faint)", display: "block" }}>Min / First</span>
                <strong style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", display: "block", whiteSpace: "nowrap" }}>{s.min}</strong>
              </div>
              <div>
                <span style={{ color: "var(--faint)", display: "block" }}>Max / Last</span>
                <strong style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", display: "block", whiteSpace: "nowrap" }}>{s.max}</strong>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
