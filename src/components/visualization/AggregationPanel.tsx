import React, { useState } from "react";
import { Table } from "@/lib/etlUtils";

interface AggregationPanelProps {
  table: Table;
}

export default function AggregationPanel({ table }: AggregationPanelProps) {
  const [groupByCol, setGroupByCol] = useState<string>(table.cols[0] || "");
  const [aggCol, setAggCol] = useState<string>(table.cols[1] || table.cols[0] || "");
  const [aggFunc, setAggFunc] = useState<"count" | "sum" | "avg" | "min" | "max">("count");

  if (!groupByCol) {
    return <div className="note">No columns available for aggregation.</div>;
  }

  const computeAggregation = () => {
    const groups: Record<string, unknown[]> = {};
    table.rows.forEach((row) => {
      const key = row[groupByCol] == null ? "—" : String(row[groupByCol]);
      if (!groups[key]) groups[key] = [];
      groups[key].push(row[aggCol]);
    });

    return Object.entries(groups).map(([groupVal, vals]) => {
      let result = 0;
      const numVals = vals.map(Number).filter((v) => !isNaN(v));

      if (aggFunc === "count") {
        result = vals.length;
      } else if (aggFunc === "sum") {
        result = numVals.reduce((a, b) => a + b, 0);
      } else if (aggFunc === "avg") {
        result = numVals.length > 0 ? numVals.reduce((a, b) => a + b, 0) / numVals.length : 0;
      } else if (aggFunc === "min") {
        result = numVals.length > 0 ? Math.min(...numVals) : 0;
      } else if (aggFunc === "max") {
        result = numVals.length > 0 ? Math.max(...numVals) : 0;
      }

      return { groupVal, result };
    });
  };

  const aggData = computeAggregation().sort((a, b) => b.result - a.result).slice(0, 10);
  const maxVal = Math.max(...aggData.map((d) => d.result)) || 1;

  return (
    <div>
      <div className="row" style={{ gap: 12, marginBottom: 20, flexWrap: "wrap", alignItems: "center" }}>
        <div className="row" style={{ gap: 6, alignItems: "center" }}>
          <label className="fld" style={{ margin: 0 }}>Group By:</label>
          <select value={groupByCol} onChange={(e) => setGroupByCol(e.target.value)}>
            {table.cols.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div className="row" style={{ gap: 6, alignItems: "center" }}>
          <label className="fld" style={{ margin: 0 }}>Function:</label>
          <select value={aggFunc} onChange={(e) => setAggFunc(e.target.value as "count" | "sum" | "avg" | "min" | "max")}>
            <option value="count">COUNT</option>
            <option value="sum">SUM</option>
            <option value="avg">AVERAGE</option>
            <option value="min">MIN</option>
            <option value="max">MAX</option>
          </select>
        </div>

        <div className="row" style={{ gap: 6, alignItems: "center" }}>
          <label className="fld" style={{ margin: 0 }}>Target Column:</label>
          <select value={aggCol} onChange={(e) => setAggCol(e.target.value)}>
            {table.cols.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {aggData.map((d, i) => {
          const barWidth = `${(d.result / maxVal) * 100}%`;
          return (
            <div key={i} style={{ display: "grid", gridTemplateColumns: "150px 1fr 80px", alignItems: "center", gap: 10 }}>
              <div style={{ textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", fontSize: 13 }} title={d.groupVal}>
                {d.groupVal}
              </div>
              <div style={{ background: "var(--border)", borderRadius: 6, height: 18, overflow: "hidden" }}>
                <div style={{ background: "var(--accent, #3b82f6)", width: barWidth, height: "100%", borderRadius: 6, transition: "width 0.3s ease" }} />
              </div>
              <div style={{ fontSize: 12, fontWeight: "bold", color: "#f3f4f6", textAlign: "right" }}>
                {d.result.toFixed(1).replace(/\.0$/, "")}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
