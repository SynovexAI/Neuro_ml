import React, { useState, Suspense } from "react";
import TableComparison from "./TableComparison";
import StatisticsPanel from "./StatisticsPanel";
import DistributionCharts from "./DistributionCharts";
import MissingValueViewer from "./MissingValueViewer";
import DataTypeViewer from "./DataTypeViewer";
import AggregationPanel from "./AggregationPanel";
import TransformationTimeline from "./TransformationTimeline";
import { Table, EtlOp, Expectation } from "@/lib/etlUtils";

interface DataVizExplorerProps {
  sourceTable: Table;
  finalTable: Table;
  ops: EtlOp[];
  rules: Expectation[];
}

const tabs = [
  "Table Comparison",
  "Statistics",
  "Distribution",
  "Missing Values",
  "Data Types",
  "Aggregation",
  "Transformation History",
] as const;

type Tab = typeof tabs[number];

export default function DataVizExplorer({ sourceTable, finalTable, ops, rules }: DataVizExplorerProps) {
  const [activeTab, setActiveTab] = useState<Tab>(tabs[0]);

  const renderTab = () => {
    switch (activeTab) {
      case "Table Comparison":
        return <TableComparison before={sourceTable} after={finalTable} />;
      case "Statistics":
        return <StatisticsPanel table={finalTable} />;
      case "Distribution":
        return <DistributionCharts table={finalTable} />;
      case "Missing Values":
        return <MissingValueViewer table={finalTable} />;
      case "Data Types":
        return <DataTypeViewer table={finalTable} />;
      case "Aggregation":
        return <AggregationPanel table={finalTable} />;
      case "Transformation History":
        return <TransformationTimeline ops={ops} sourceTable={sourceTable} finalTable={finalTable} />;
      default:
        return null;
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div className="seg" style={{ flexWrap: "wrap" }}>
        {tabs.map((t) => (
          <button
            key={t}
            className={activeTab === t ? "on" : ""}
            onClick={() => setActiveTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <div style={{ padding: "8px 0" }}>
        <Suspense fallback={<div style={{ textAlign: "center", padding: 20, color: "var(--faint)" }}>Loading visualization...</div>}>
          {renderTab()}
        </Suspense>
      </div>
    </div>
  );
}
