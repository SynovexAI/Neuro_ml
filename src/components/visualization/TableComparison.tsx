import React from "react";
import { Table } from "@/lib/etlUtils";

interface TableComparisonProps {
  before: Table;
  after: Table;
}

export default function TableComparison({ before, after }: TableComparisonProps) {
  const beforeRows = before.rows.length;
  const afterRows = after.rows.length;
  const maxRows = Math.max(beforeRows, afterRows, 1);

  const beforeCols = before.cols.length;
  const afterCols = after.cols.length;
  const maxCols = Math.max(beforeCols, afterCols, 1);

  // Identify added, removed, and kept columns
  const beforeSet = new Set(before.cols);
  const afterSet = new Set(after.cols);
  
  const keptCols = before.cols.filter(c => afterSet.has(c));
  const removedCols = before.cols.filter(c => !afterSet.has(c));
  const addedCols = after.cols.filter(c => !beforeSet.has(c));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Visual metric comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Row count comparison */}
        <div style={{ background: "rgba(255,255,255,0.02)", padding: 16, borderRadius: 8, border: "1px solid var(--border)" }}>
          <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: "var(--faint)" }}>Row Count Comparison</h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span>Before (Source)</span>
                <strong>{beforeRows} rows</strong>
              </div>
              <div style={{ background: "var(--border)", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ background: "#3b82f6", width: `${(beforeRows / maxRows) * 100}%`, height: "100%", borderRadius: 4 }} />
              </div>
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span>After (Output)</span>
                <strong>{afterRows} rows</strong>
              </div>
              <div style={{ background: "var(--border)", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ background: "#10b981", width: `${(afterRows / maxRows) * 100}%`, height: "100%", borderRadius: 4 }} />
              </div>
            </div>
          </div>
        </div>

        {/* Column count comparison */}
        <div style={{ background: "rgba(255,255,255,0.02)", padding: 16, borderRadius: 8, border: "1px solid var(--border)" }}>
          <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: "var(--faint)" }}>Column Count Comparison</h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span>Before (Source)</span>
                <strong>{beforeCols} columns</strong>
              </div>
              <div style={{ background: "var(--border)", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ background: "#a855f7", width: `${(beforeCols / maxCols) * 100}%`, height: "100%", borderRadius: 4 }} />
              </div>
            </div>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span>After (Output)</span>
                <strong>{afterCols} columns</strong>
              </div>
              <div style={{ background: "var(--border)", height: 8, borderRadius: 4, overflow: "hidden" }}>
                <div style={{ background: "#ec4899", width: `${(afterCols / maxCols) * 100}%`, height: "100%", borderRadius: 4 }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Schema comparison visualization */}
      <div style={{ background: "rgba(255,255,255,0.02)", padding: 16, borderRadius: 8, border: "1px solid var(--border)" }}>
        <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12, color: "var(--faint)" }}>Schema Evolution</h4>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {removedCols.map(c => (
            <span key={c} style={{ padding: "4px 8px", borderRadius: 6, fontSize: 12, background: "rgba(239, 68, 68, 0.15)", border: "1px solid rgba(239, 68, 68, 0.3)", color: "#f87171" }}>
              ➖ {c} (Removed)
            </span>
          ))}
          {keptCols.map(c => (
            <span key={c} style={{ padding: "4px 8px", borderRadius: 6, fontSize: 12, background: "rgba(255, 255, 255, 0.05)", border: "1px solid var(--border)", color: "#e5e7eb" }}>
              ➡️ {c} (Kept)
            </span>
          ))}
          {addedCols.map(c => (
            <span key={c} style={{ padding: "4px 8px", borderRadius: 6, fontSize: 12, background: "rgba(16, 185, 129, 0.15)", border: "1px solid rgba(16, 185, 129, 0.3)", color: "#34d399" }}>
              ➕ {c} (Added)
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
