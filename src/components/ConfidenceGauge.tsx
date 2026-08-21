"use client";

import React from "react";
import type { ConfidenceMetrics } from "@/lib/agentEval";

interface ConfidenceGaugeProps {
  metrics: ConfidenceMetrics;
  size?: number;
  showBreakdown?: boolean;
  compact?: boolean;
}

export default function ConfidenceGauge({
  metrics,
  size = 96,
  showBreakdown = true,
  compact = false,
}: ConfidenceGaugeProps) {
  const { score, grounding, toolReliability, reasoningConsistency, factualDensity, label, explanation } = metrics;

  const strokeWidth = compact ? 7 : 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  // Dynamic colors
  const color =
    score >= 80 ? "var(--good, #10b981)" : score >= 60 ? "#f59e0b" : "#f43f5e";
  const bgTrack = "rgba(255, 255, 255, 0.08)";

  if (compact) {
    return (
      <div style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
        <div style={{ position: "relative", width: size, height: size }}>
          <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={bgTrack}
              strokeWidth={strokeWidth}
              fill="transparent"
            />
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)" }}
            />
          </svg>
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 800,
              fontSize: size * 0.28,
              color: "var(--text, #fff)",
              fontFamily: "var(--mono, monospace)",
            }}
          >
            {score}%
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <span style={{ fontSize: 12, fontWeight: 700, color }}>{label}</span>
          <span style={{ fontSize: 10, color: "var(--muted, #8b949e)" }}>{score}/100 confidence</span>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        padding: "14px 16px",
        background: "rgba(255, 255, 255, 0.02)",
        border: "1px solid var(--border)",
        borderRadius: 12,
        marginTop: 12,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
        {/* SVG Circle Gauge */}
        <div style={{ position: "relative", width: size, height: size, flexShrink: 0 }}>
          <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={bgTrack}
              strokeWidth={strokeWidth}
              fill="transparent"
            />
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              stroke={color}
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)" }}
            />
          </svg>
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontWeight: 800,
                fontSize: size * 0.28,
                color: "var(--text, #fff)",
                fontFamily: "var(--mono, monospace)",
                lineHeight: 1,
              }}
            >
              {score}%
            </span>
            <span style={{ fontSize: 9, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginTop: 2 }}>
              Score
            </span>
          </div>
        </div>

        {/* Overview Header & Summary */}
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span
              style={{
                fontSize: 13,
                fontWeight: 700,
                color,
                background: `color-mix(in srgb, ${color} 15%, transparent)`,
                padding: "2px 8px",
                borderRadius: 6,
                border: `1px solid color-mix(in srgb, ${color} 30%, transparent)`,
              }}
            >
              🎯 {label}
            </span>
            <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)" }}>
              {score}/100 composite index
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary, #c9d1d9)", margin: "4px 0 0", lineHeight: 1.45 }}>
            {explanation}
          </p>
        </div>
      </div>

      {/* Breakdown Metrics Grid */}
      {showBreakdown && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
            gap: 8,
            marginTop: 4,
            paddingTop: 10,
            borderTop: "1px solid rgba(255, 255, 255, 0.05)",
          }}
        >
          <MetricBar label="Fact Grounding" value={grounding} hint="Tool evidence match" />
          <MetricBar label="Tool Reliability" value={toolReliability} hint="Execution success" />
          <MetricBar label="Reasoning Flow" value={reasoningConsistency} hint="Loop convergence" />
          <MetricBar label="Fact Density" value={factualDensity} hint="Concrete data count" />
        </div>
      )}
    </div>
  );
}

function MetricBar({ label, value, hint }: { label: string; value: number; hint: string }) {
  const color = value >= 75 ? "var(--good, #10b981)" : value >= 50 ? "#f59e0b" : "#f43f5e";

  return (
    <div
      style={{
        background: "rgba(255, 255, 255, 0.02)",
        border: "1px solid rgba(255, 255, 255, 0.06)",
        borderRadius: 8,
        padding: "6px 8px",
        display: "flex",
        flexDirection: "column",
        gap: 3,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
        <span style={{ color: "var(--muted)", fontWeight: 500 }}>{label}</span>
        <span style={{ fontFamily: "var(--mono)", fontWeight: 700, color }}>{value}%</span>
      </div>
      <div style={{ height: 4, width: "100%", background: "rgba(255, 255, 255, 0.08)", borderRadius: 2, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${value}%`,
            background: color,
            borderRadius: 2,
            transition: "width 0.6s ease",
          }}
        />
      </div>
      <span style={{ fontSize: 9, color: "var(--muted)", opacity: 0.75 }}>{hint}</span>
    </div>
  );
}
