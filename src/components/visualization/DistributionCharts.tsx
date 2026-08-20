import React, { useState } from "react";
import { Table } from "@/lib/etlUtils";

interface DistributionChartsProps {
  table: Table;
}

export default function DistributionCharts({ table }: DistributionChartsProps) {
  const [selectedCol, setSelectedCol] = useState<string>(table.cols[0] || "");

  if (!selectedCol) {
    return <div className="note">No columns available.</div>;
  }

  const getDistribution = (col: string) => {
    const values = table.rows.map((r) => r[col]).filter((v) => v !== null && v !== undefined);
    
    // Check if numeric to bin it
    const numValues = values.map((v) => Number(v)).filter((v) => !isNaN(v));
    const isNumeric = numValues.length > 0 && numValues.length === values.length;

    if (isNumeric) {
      const min = Math.min(...numValues);
      const max = Math.max(...numValues);
      const range = max - min;
      const binCount = 5;
      const binSize = range / binCount || 1;
      const bins = Array.from({ length: binCount }, (_, i) => {
        const start = min + i * binSize;
        const end = start + binSize;
        return {
          label: `${start.toFixed(1)} - ${end.toFixed(1)}`,
          count: 0,
          start,
          end,
        };
      });

      numValues.forEach((val) => {
        let placed = false;
        for (let i = 0; i < binCount; i++) {
          if (val >= bins[i].start && val <= bins[i].end) {
            bins[i].count++;
            placed = true;
            break;
          }
        }
        if (!placed && val >= bins[binCount - 1].end) {
          bins[binCount - 1].count++;
        }
      });

      return bins;
    } else {
      // Categorical frequency
      const counts: Record<string, number> = {};
      values.forEach((v) => {
        const str = String(v);
        counts[str] = (counts[str] || 0) + 1;
      });

      return Object.entries(counts)
        .map(([label, count]) => ({ label, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 10); // top 10 values
    }
  };

  const dist = getDistribution(selectedCol);
  const total = dist.reduce((acc, curr) => acc + curr.count, 0) || 1;
  const maxCount = Math.max(...dist.map((d) => d.count)) || 1;

  return (
    <div>
      <div className="row" style={{ gap: 10, marginBottom: 15, alignItems: "center" }}>
        <label className="fld" style={{ margin: 0 }}>Select Column:</label>
        <select value={selectedCol} onChange={(e) => setSelectedCol(e.target.value)} style={{ maxWidth: 200 }}>
          {table.cols.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {dist.map((d, i) => {
          const pct = ((d.count / total) * 100).toFixed(1);
          const barWidth = `${(d.count / maxCount) * 100}%`;
          return (
            <div key={i} style={{ display: "grid", gridTemplateColumns: "150px 1fr 60px", alignItems: "center", gap: 10 }}>
              <div style={{ textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", fontSize: 13 }} title={d.label}>
                {d.label}
              </div>
              <div style={{ background: "var(--border)", borderRadius: 6, height: 18, overflow: "hidden", position: "relative" }}>
                <div style={{ background: "var(--accent, #ec4899)", width: barWidth, height: "100%", borderRadius: 6, transition: "width 0.3s ease" }} />
              </div>
              <div style={{ fontSize: 12, color: "var(--faint)", textAlign: "right" }}>
                {d.count} ({pct}%)
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
