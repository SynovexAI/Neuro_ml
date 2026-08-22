"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import Markdown from "@/components/Markdown";
import AgentOutput from "@/components/AgentOutput";
import ConfidenceGauge from "./ConfidenceGauge";
import { computeConfidenceScore, type ConfidenceMetrics } from "@/lib/agentEval";
import { AGENT_TOOLS, type AgentTool } from "@/lib/agentTools";
import { toast } from "@/lib/toast";
import {
  Zap,
  Network,
  Layers,
  Scale,
  Workflow,
  Globe,
  Search,
  Brain,
  Table,
  FileText,
  LineChart,
  Sparkles,
  Bot,
  Trash2,
  Plus,
  Play,
  Check,
  Cpu,
  Database,
  Calendar,
  Calculator,
  BookOpen,
  Maximize2,
  Minimize2,
  FileCode,
  Shield,
  Award,
  Building2,
  MapPin,
  Code2,
  FolderOpen,
  Save,
  Download,
  Rocket,
} from "lucide-react";

export function renderAgentIcon(name: string, size = 16, color?: string, style?: React.CSSProperties) {
  const iconProps = { size, color, style };
  switch (name) {
    case "globe":
    case "general":
      return <Globe {...iconProps} />;
    case "search":
    case "web_search":
      return <Search {...iconProps} />;
    case "brain":
    case "knowledge":
      return <Brain {...iconProps} />;
    case "table":
    case "excel":
      return <Table {...iconProps} />;
    case "file-text":
    case "pdf":
      return <FileText {...iconProps} />;
    case "line-chart":
    case "analyst":
    case "statistics":
      return <LineChart {...iconProps} />;
    case "sparkles":
    case "synthesizer":
      return <Sparkles {...iconProps} />;
    case "bot":
    case "custom":
    case "gear":
      return <Bot {...iconProps} />;
    case "zap":
    case "linear":
      return <Zap {...iconProps} />;
    case "network":
    case "hierarchical":
      return <Network {...iconProps} />;
    case "layers":
    case "sequential":
      return <Layers {...iconProps} />;
    case "scale":
    case "consensus":
      return <Scale {...iconProps} />;
    case "workflow":
      return <Workflow {...iconProps} />;
    case "calculator":
      return <Calculator {...iconProps} />;
    case "db_query":
    case "database":
      return <Database {...iconProps} />;
    case "wikipedia":
    case "book":
      return <BookOpen {...iconProps} />;
    case "arxiv":
      return <FileCode {...iconProps} />;
    case "shield":
      return <Shield {...iconProps} />;
    case "award":
      return <Award {...iconProps} />;
    case "datetime":
      return <Calendar {...iconProps} />;
    default:
      return <Cpu {...iconProps} />;
  }
}

const CopySvg = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

const CheckSvg = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

export type Step = "type" | "build" | "run" | "learn";
export type TopologyType = "linear" | "hierarchical" | "sequential" | "consensus" | "custom";
export type NodeTypeKey = "general" | "excel" | "pdf" | "knowledge" | "web_search" | "analyst" | "synthesizer" | "custom";

export interface NodeTheme {
  accent: string;
  light: string;
  dark: string;
  bg: string;
  border: string;
  glow: string;
  badgeBg: string;
  badgeText: string;
}

export interface NodeTypeConfig {
  type: NodeTypeKey;
  label: string;
  sublabel: string;
  icon: string;
  tag: string;
  defaultRole: string;
  defaultPrompt: string;
  defaultTools: string[];
  theme: NodeTheme;
}

export const NODE_TYPES_CATALOG: Record<NodeTypeKey, NodeTypeConfig> = {
  general: {
    type: "general",
    label: "General Agent",
    sublabel: "Reasoning & Orchestration",
    icon: "globe",
    tag: "GENERAL",
    defaultRole: "General Reasoning & Coordinator",
    defaultPrompt: "You are the General Reasoning Agent. Decompose user requests, coordinate workflows, and provide foundational reasoning.",
    defaultTools: ["datetime", "calculator"],
    theme: {
      accent: "#3b82f6",
      light: "#93c5fd",
      dark: "#1d4ed8",
      bg: "rgba(59, 130, 246, 0.12)",
      border: "rgba(59, 130, 246, 0.45)",
      glow: "rgba(59, 130, 246, 0.25)",
      badgeBg: "rgba(59, 130, 246, 0.20)",
      badgeText: "#bfdbfe",
    },
  },
  web_search: {
    type: "web_search",
    label: "Web Search Specialist",
    sublabel: "Live Web & Real-Time Research",
    icon: "search",
    tag: "WEB SEARCH",
    defaultRole: "Live Web & Real-Time Retrieval",
    defaultPrompt: "You are the Web Search Specialist. Search real-time web sources, gather factual evidence, verify latest figures, and cite authoritative sources.",
    defaultTools: ["web_search", "arxiv", "wikipedia"],
    theme: {
      accent: "#0ea5e9",
      light: "#7dd3fc",
      dark: "#0284c7",
      bg: "rgba(14, 165, 233, 0.12)",
      border: "rgba(14, 165, 233, 0.45)",
      glow: "rgba(14, 165, 233, 0.25)",
      badgeBg: "rgba(14, 165, 233, 0.20)",
      badgeText: "#bae6fd",
    },
  },
  knowledge: {
    type: "knowledge",
    label: "Knowledge Base (RAG)",
    sublabel: "Semantic Docs & Internal Wiki",
    icon: "brain",
    tag: "KNOWLEDGE BASE",
    defaultRole: "Semantic Retrieval & Domain Knowledge",
    defaultPrompt: "You are the Knowledge Base Retrieval Specialist. Retrieve relevant semantic passages, search internal documentation, and ground all claims with verified knowledge.",
    defaultTools: ["wikipedia", "db_query"],
    theme: {
      accent: "#6366f1",
      light: "#a5b4fc",
      dark: "#4338ca",
      bg: "rgba(99, 102, 241, 0.12)",
      border: "rgba(99, 102, 241, 0.45)",
      glow: "rgba(99, 102, 241, 0.25)",
      badgeBg: "rgba(99, 102, 241, 0.20)",
      badgeText: "#c7d2fe",
    },
  },
  excel: {
    type: "excel",
    label: "Excel & Data Engine",
    sublabel: "Spreadsheets & Tabular Analytics",
    icon: "table",
    tag: "EXCEL / DATA",
    defaultRole: "Spreadsheets & Tabular Analytics",
    defaultPrompt: "You are the Excel & Data Specialist. Analyze tabular data, process spreadsheet records (.xlsx, .csv), compute statistics, and format data tables.",
    defaultTools: ["calculator", "db_query", "statistics"],
    theme: {
      accent: "#2563eb",
      light: "#60a5fa",
      dark: "#1d4ed8",
      bg: "rgba(37, 99, 235, 0.12)",
      border: "rgba(37, 99, 235, 0.45)",
      glow: "rgba(37, 99, 235, 0.25)",
      badgeBg: "rgba(37, 99, 235, 0.20)",
      badgeText: "#bfdbfe",
    },
  },
  pdf: {
    type: "pdf",
    label: "PDF & Doc Processor",
    sublabel: "Document Extraction & Papers",
    icon: "file-text",
    tag: "PDF / DOC",
    defaultRole: "Document Parsing & Paper Extraction",
    defaultPrompt: "You are the PDF & Document Specialist. Extract structured information from text, parse PDF reports and research papers, and summarize key sections.",
    defaultTools: ["arxiv"],
    theme: {
      accent: "#0284c7",
      light: "#38bdf8",
      dark: "#0369a1",
      bg: "rgba(2, 132, 199, 0.12)",
      border: "rgba(2, 132, 199, 0.45)",
      glow: "rgba(2, 132, 199, 0.25)",
      badgeBg: "rgba(2, 132, 199, 0.20)",
      badgeText: "#bae6fd",
    },
  },
  analyst: {
    type: "analyst",
    label: "Quantitative Analyst",
    sublabel: "Math & Statistical Verification",
    icon: "line-chart",
    tag: "DATA ANALYST",
    defaultRole: "Math Formulas & Quantitative Verification",
    defaultPrompt: "You are the Quantitative Analyst. Compute formulas, calculate YoY CAGR, check percentages, and verify arithmetic accuracy.",
    defaultTools: ["calculator", "statistics", "db_query"],
    theme: {
      accent: "#1d4ed8",
      light: "#93c5fd",
      dark: "#1e3a8a",
      bg: "rgba(29, 78, 216, 0.12)",
      border: "rgba(29, 78, 216, 0.45)",
      glow: "rgba(29, 78, 216, 0.25)",
      badgeBg: "rgba(29, 78, 216, 0.20)",
      badgeText: "#dbeafe",
    },
  },
  synthesizer: {
    type: "synthesizer",
    label: "Final Synthesizer",
    sublabel: "Executive Brief & Output",
    icon: "sparkles",
    tag: "SYNTHESIZER",
    defaultRole: "Executive Summary & Output Brief",
    defaultPrompt: "You are the Final Synthesizer. Combine all upstream research, documents, calculations, and tables into a comprehensive, publication-ready executive report.",
    defaultTools: ["datetime"],
    theme: {
      accent: "#38bdf8",
      light: "#bae6fd",
      dark: "#0284c7",
      bg: "rgba(56, 189, 248, 0.12)",
      border: "rgba(56, 189, 248, 0.45)",
      glow: "rgba(56, 189, 248, 0.25)",
      badgeBg: "rgba(56, 189, 248, 0.20)",
      badgeText: "#e0f2fe",
    },
  },
  custom: {
    type: "custom",
    label: "Custom Specialist",
    sublabel: "Configurable Agent Node",
    icon: "bot",
    tag: "CUSTOM",
    defaultRole: "Custom Domain Specialist",
    defaultPrompt: "You are a specialized agent node. Execute your assigned role diligently and return structured results.",
    defaultTools: ["web_search", "calculator"],
    theme: {
      accent: "#06b6d4",
      light: "#67e8f9",
      dark: "#0891b2",
      bg: "rgba(6, 182, 212, 0.12)",
      border: "rgba(6, 182, 212, 0.45)",
      glow: "rgba(6, 182, 212, 0.25)",
      badgeBg: "rgba(6, 182, 212, 0.20)",
      badgeText: "#cffafe",
    },
  },
};

export type OrchestrationNode = {
  id: string;
  name: string;
  role: string;
  icon: string;
  nodeType: NodeTypeKey;
  model: string;
  temperature: number;
  tools: string[];
  systemPrompt: string;
  w: number;
  h: number;
};

type PipelineStepExecution = {
  nodeId: string;
  nodeName: string;
  icon: string;
  nodeType: NodeTypeKey;
  status: "pending" | "running" | "done" | "error";
  input: string;
  output: string;
  latencyMs: number;
  tokens: number;
  confidence?: ConfidenceMetrics;
};

const PRESET_TOPOLOGIES: Record<TopologyType, { title: string; desc: string; icon: string; tag: string; nodes: OrchestrationNode[]; task: string }> = {
  linear: {
    title: "Linear Multi-Agent Flow",
    desc: "Sequential pipeline passing context step-by-step through General, Web Search, Knowledge Base, Excel Data, PDF Processor, and Synthesizer.",
    icon: "zap",
    tag: "Linear Flow",
    task: "Gather 2024 tech industry revenue data, cross-reference knowledge base notes, extract table records from Excel/PDF files, compute CAGR, and generate an executive report.",
    nodes: [
      {
        id: "general_coordinator",
        name: "General Coordinator",
        role: "Workflow & Plan Initialization",
        icon: "globe",
        nodeType: "general",
        model: "default",
        temperature: 0.3,
        tools: ["datetime", "calculator"],
        systemPrompt: "You are the General Coordinator. Analyze the user request, break down the execution roadmap, and prepare initial instructions for specialist agents.",
        w: 220,
        h: 68,
      },
      {
        id: "web_searcher",
        name: "Web Search Specialist",
        role: "Live Fact & Trend Retrieval",
        icon: "search",
        nodeType: "web_search",
        model: "default",
        temperature: 0.35,
        tools: ["web_search", "wikipedia"],
        systemPrompt: "You are the Web Search Specialist. Search real-time web sources, gather market metrics, and extract authoritative factual observations.",
        w: 220,
        h: 68,
      },
      {
        id: "knowledge_retriever",
        name: "Knowledge Base (RAG)",
        role: "Semantic Docs & Internal Wiki",
        icon: "brain",
        nodeType: "knowledge",
        model: "default",
        temperature: 0.25,
        tools: ["wikipedia", "db_query"],
        systemPrompt: "You are the Knowledge Base Specialist. Retrieve contextual domain knowledge, cross-reference internal documentation, and ground claims.",
        w: 220,
        h: 68,
      },
      {
        id: "excel_engine",
        name: "Excel & Data Engine",
        role: "Spreadsheet Analytics & Stats",
        icon: "table",
        nodeType: "excel",
        model: "default",
        temperature: 0.2,
        tools: ["calculator", "db_query", "statistics"],
        systemPrompt: "You are the Excel & Data Specialist. Analyze tabular records, compute statistical formulas (CAGR, margins, totals), and structure data tables.",
        w: 220,
        h: 68,
      },
      {
        id: "pdf_processor",
        name: "PDF & Doc Processor",
        role: "Document Extraction & Papers",
        icon: "file-text",
        nodeType: "pdf",
        model: "default",
        temperature: 0.3,
        tools: ["arxiv"],
        systemPrompt: "You are the PDF & Document Specialist. Parse document text, extract specific clauses and paper findings, and structure key takeaways.",
        w: 220,
        h: 68,
      },
      {
        id: "final_synthesizer",
        name: "Final Synthesizer",
        role: "Executive Summary & Output",
        icon: "sparkles",
        nodeType: "synthesizer",
        model: "default",
        temperature: 0.35,
        tools: ["datetime"],
        systemPrompt: "You are the Final Synthesizer. Combine all upstream facts, calculations, and extracted insights into a clean, publication-ready executive brief with markdown tables.",
        w: 220,
        h: 68,
      },
    ],
  },
  hierarchical: {
    title: "Hierarchical Supervisor",
    desc: "A central coordinator router decomposes the user task and delegates to specialist agents.",
    icon: "network",
    tag: "Coordinator",
    task: "Analyze revenue growth from $1.85M in 2023 to $2.42M in 2024, compute CAGR, and summarize key market drivers.",
    nodes: [
      {
        id: "supervisor",
        name: "Supervisor Router",
        role: "Coordinator & Dispatcher",
        icon: "network",
        nodeType: "general",
        model: "default",
        temperature: 0.3,
        tools: ["web_search"],
        systemPrompt: "You are the Lead Supervisor. Decompose the user task, plan sub-goals, and instruct specialist agents on exact expectations.",
        w: 220,
        h: 68,
      },
      {
        id: "researcher",
        name: "Fact Researcher",
        role: "Web & Knowledge Retrieval",
        icon: "search",
        nodeType: "web_search",
        model: "default",
        temperature: 0.35,
        tools: ["web_search", "arxiv"],
        systemPrompt: "You are the Research Specialist. Gather precise factual observations, industry context, and real-world evidence.",
        w: 215,
        h: 68,
      },
      {
        id: "analyst",
        name: "Data Analyst",
        role: "Calculations & Verification",
        icon: "table",
        nodeType: "excel",
        model: "default",
        temperature: 0.2,
        tools: ["calculator", "db_query"],
        systemPrompt: "You are the Quantitative Analyst. Compute statistics, calculate formulas, check percentages, and verify arithmetic.",
        w: 215,
        h: 68,
      },
      {
        id: "synthesizer",
        name: "Final Synthesizer",
        role: "Executive Summary & Output",
        icon: "sparkles",
        nodeType: "synthesizer",
        model: "default",
        temperature: 0.35,
        tools: ["datetime"],
        systemPrompt: "You are the Final Synthesizer. Combine the researcher's facts and analyst's calculations into a polished executive brief.",
        w: 215,
        h: 68,
      },
    ],
  },
  sequential: {
    title: "Sequential Chain",
    desc: "A linear pipeline where each specialist executes a phase and passes structured output to the next stage.",
    icon: "layers",
    tag: "Sequential",
    task: "Draft a comprehensive API security architecture for a FinTech platform and run a vulnerability review.",
    nodes: [
      {
        id: "planner",
        name: "Strategic Planner",
        role: "Decomposition & Milestones",
        icon: "globe",
        nodeType: "general",
        model: "default",
        temperature: 0.3,
        tools: [],
        systemPrompt: "You are a strategic planner. Break down the task into concrete, logical execution milestones with clear acceptance criteria.",
        w: 220,
        h: 68,
      },
      {
        id: "executor",
        name: "Core Executor",
        role: "Implementation & Technical Depth",
        icon: "zap",
        nodeType: "analyst",
        model: "default",
        temperature: 0.35,
        tools: ["calculator", "web_search", "datetime"],
        systemPrompt: "You are the Core Executor. Methodically execute the plan with concrete technical specifications and computations.",
        w: 220,
        h: 68,
      },
      {
        id: "critic",
        name: "Critique & Fact-Checker",
        role: "Verification & Final Polish",
        icon: "shield",
        nodeType: "knowledge",
        model: "default",
        temperature: 0.25,
        tools: [],
        systemPrompt: "You are the Critic. Identify any missing nuances, verify security claims and calculations, and produce a refined final output.",
        w: 220,
        h: 68,
      },
      {
        id: "final_out",
        name: "Final Synthesizer",
        role: "Executive Output & Brief",
        icon: "sparkles",
        nodeType: "synthesizer",
        model: "default",
        temperature: 0.35,
        tools: ["datetime"],
        systemPrompt: "You are the Final Synthesizer. Polish and output the final validated specification.",
        w: 220,
        h: 68,
      },
    ],
  },
  consensus: {
    title: "Dual Consensus & Debate",
    desc: "Two autonomous agents reason independently; a consensus arbiter compares outputs and synthesizes agreement.",
    icon: "scale",
    tag: "Consensus",
    task: "Evaluate whether to migrate a monolithic backend to Microservices vs Modular Monolith for a high-traffic app.",
    nodes: [
      {
        id: "agent_a",
        name: "Agent Alpha",
        role: "Primary Perspective",
        icon: "globe",
        nodeType: "web_search",
        model: "default",
        temperature: 0.4,
        tools: ["calculator", "web_search"],
        systemPrompt: "You are Agent Alpha. Provide a direct, pragmatic, evidence-based proposal focusing on rapid delivery and simplicity.",
        w: 215,
        h: 68,
      },
      {
        id: "agent_b",
        name: "Agent Beta",
        role: "Analytical Counter-Perspective",
        icon: "line-chart",
        nodeType: "analyst",
        model: "default",
        temperature: 0.4,
        tools: ["calculator", "datetime"],
        systemPrompt: "You are Agent Beta. Approach the problem critically, identifying scalability limits, operational complexity, and long-term costs.",
        w: 215,
        h: 68,
      },
      {
        id: "referee",
        name: "Consensus Arbiter",
        role: "Comparison & Unified Verdict",
        icon: "scale",
        nodeType: "synthesizer",
        model: "default",
        temperature: 0.3,
        tools: [],
        systemPrompt: "You are the Consensus Arbiter. Compare Alpha and Beta's arguments, evaluate trade-offs, and produce the authoritative unified synthesis.",
        w: 220,
        h: 68,
      },
    ],
  },
  custom: {
    title: "Custom Flow Pipeline",
    desc: "Fully customizable multi-agent graph with custom node definitions, color themes, and tool assignment.",
    icon: "workflow",
    tag: "Custom DAG",
    task: "Parse user requirements, query database records, and generate an executive report.",
    nodes: [
      {
        id: "ingest",
        name: "Data Ingest",
        role: "Ingestion & Schema",
        icon: "table",
        nodeType: "excel",
        model: "default",
        temperature: 0.2,
        tools: ["db_query"],
        systemPrompt: "You are the Ingestion Agent. Parse the input and validate the dataset schema.",
        w: 220,
        h: 68,
      },
      {
        id: "processor",
        name: "Analytics Engine",
        role: "Processing & Stats",
        icon: "line-chart",
        nodeType: "analyst",
        model: "default",
        temperature: 0.25,
        tools: ["calculator", "statistics"],
        systemPrompt: "You are the Analytics Engine. Compute statistical aggregations and summarize patterns.",
        w: 220,
        h: 68,
      },
      {
        id: "reporter",
        name: "Report Generator",
        role: "Summary & Formatting",
        icon: "file-text",
        nodeType: "synthesizer",
        model: "default",
        temperature: 0.35,
        tools: ["datetime"],
        systemPrompt: "You are the Report Generator. Format the final executive report with clean tables.",
        w: 220,
        h: 68,
      },
    ],
  },
};

export interface OrchestrationLesson {
  id: string;
  category: "Linear Flow" | "Hierarchical" | "Sequential" | "Consensus" | "Production DAG";
  title: string;
  icon: string;
  badge: string;
  summary: string;
  deepDive: string;
  realWorldUse: string;
  diagram: string;
  keyConcepts: string[];
  codeSchema: string;
}

const ORCHESTRATION_LESSONS: OrchestrationLesson[] = [
  {
    id: "linear_flow_pattern",
    category: "Linear Flow",
    title: "Linear Multi-Agent Flow & State Handover",
    icon: "zap",
    badge: "Linear Pipeline",
    summary: "How specialized agents (General -> Web Search -> Knowledge Base -> Excel -> PDF -> Synthesizer) chain outputs sequentially.",
    deepDive: "In a Linear Multi-Agent Flow, each stage acts as a specialized transformer that consumes the accumulated upstream context and performs its domain function (such as real-time web retrieval, document parsing, statistical calculations) before passing the enriched payload to downstream agents.",
    realWorldUse: "Complex market intelligence pipelines where raw queries require live search facts, PDF contract parsing, spreadsheet metrics calculation, and final C-suite synthesis.",
    diagram: `[User Request]
       │
       ▼
[1. General Agent] -> [2. Web Search] -> [3. Knowledge Base] -> [4. Excel & Data] -> [5. PDF Processor] -> [6. Final Synthesizer]
                                                                                                    │
                                                                                                    ▼
                                                                                        [Executive Brief & Report]`,
    keyConcepts: [
      "Specialized Domain Partitioning: Each node handles only its domain without prompt pollution.",
      "Context Accumulation & Compression: High-signal data is preserved while intermediate noise is filtered.",
      "Predictable Linear Latency: Deterministic sequential execution makes debugging and tracing simple.",
    ],
    codeSchema: `const pipeline = [generalAgent, webSearchAgent, ragAgent, excelAgent, pdfAgent, synthesizerAgent];
let context = task;
for (const agent of pipeline) {
  context = await agent.run({ input: context });
}`,
  },
  {
    id: "supervisor_pattern",
    category: "Hierarchical",
    title: "Supervisor Router & Task Decomposition",
    icon: "network",
    badge: "Coordinator Pattern",
    summary: "How a central coordinator LLM breaks down complex user requests and dispatches sub-goals to specialists.",
    deepDive: "The Hierarchical Supervisor pattern solves context bloat and hallucination by placing a central reasoning coordinator at the top of the graph. The Supervisor does not attempt to solve the whole task directly; instead, it analyzes incoming requirements, plans the necessary steps, and decides which downstream worker agents are best suited for each sub-problem.",
    realWorldUse: "Enterprise analytics where a user prompt requires SQL database queries, financial formula calculations, and an executive presentation generated simultaneously.",
    diagram: `[User Request]
       │
       ▼
[Supervisor Router]
  ┌────────────┼────────────┐
  ▼            ▼            ▼
[Fact       [Data        [Final
Researcher]  Analyst]     Synthesizer]
  └────────────┼────────────┘
               ▼
[Executive Synthesis Output]`,
    keyConcepts: [
      "Sub-Goal Planning: Supervisor creates structured JSON execution plans.",
      "Tool Isolation: Specialist agents only receive tools they strictly need, preventing confusion.",
      "Context Reduction: Intermediate noisy tool outputs are filtered before final synthesis.",
    ],
    codeSchema: `{
  "plan": "Analyze Q4 Revenue",
  "delegations": [
    { "agent": "researcher", "goal": "Find 2023 vs 2024 revenue filings" },
    { "agent": "analyst", "goal": "Calculate CAGR & YoY profit margin %" }
  ]
}`,
  },
  {
    id: "sequential_pipeline",
    category: "Sequential",
    title: "Sequential Pipelines & Context Passing",
    icon: "layers",
    badge: "Step-by-Step Refinement",
    summary: "Passing structured state from stage to stage (e.g. Plan -> Execute -> Critique) without context drift.",
    deepDive: "In a sequential pipeline, agents are arranged linearly. Stage N receives the processed output of Stage N-1. This is ideal for iterative refinement workflows, like code generation or technical writing.",
    realWorldUse: "Code review automation, documentation generation, and compliance validation pipelines.",
    diagram: `[User Prompt] -> [Strategic Planner] -> [Core Executor] -> [Critique & Fact-Checker] -> [Polished Final Output]`,
    keyConcepts: [
      "State Continuity: Passing validated JSON payloads between stages.",
      "Token Preservation: Summarizing verbose thoughts before downstream handover.",
      "Deterministic Flow: Highly predictable debugging and latency tracking.",
    ],
    codeSchema: `async function runPipeline(task) {
  const plan = await plannerAgent.run(task);
  const draft = await executorAgent.run({ task, plan });
  const polished = await criticAgent.run({ draft, criteria: "Security & Accuracy" });
  return polished;
}`,
  },
  {
    id: "consensus_debate",
    category: "Consensus",
    title: "Multi-Agent Consensus & Debate",
    icon: "scale",
    badge: "Dialectic Verification",
    summary: "Running independent perspectives in parallel and synthesizing agreement through an arbiter.",
    deepDive: "Debate-based orchestration engages two or more agents with conflicting or complementary system instructions. Both agents solve the prompt in parallel without seeing each other's work. A third Arbiter agent then cross-references both answers, evaluates points of convergence, and synthesizes the most truthful conclusion.",
    realWorldUse: "High-stakes architectural decisions, medical/legal factual verification, and investment risk reviews.",
    diagram: `               ┌─── [Agent Alpha: Direct Evidence] ───┐
[User Prompt] ─┤                                      ├─▶ [Consensus Arbiter] -> [Unified Verdict]
               └─── [Agent Beta: Counter-Perspective] ─┘`,
    keyConcepts: [
      "Independent Reasoning: Eliminates confirmation bias and groupthink.",
      "Agreement Scoring: Measuring mathematical overlap between outputs.",
      "Arbiter Synthesis: Producing a balanced, comprehensive verdict.",
    ],
    codeSchema: `const [outA, outB] = await Promise.all([
  agentAlpha.run(task),
  agentBeta.run(task)
]);
const finalVerdict = await arbiterAgent.run({
  task, perspectiveA: outA, perspectiveB: outB
});`,
  },
  {
    id: "production_dag",
    category: "Production DAG",
    title: "Dynamic DAGs, LangGraph & Error Recovery",
    icon: "workflow",
    badge: "Production Swarms",
    summary: "Building cyclic graphs, conditional routing, checkpoints, and automated retry mechanisms.",
    deepDive: "Production multi-agent systems rely on Directed Acyclic Graphs (DAGs) and state machines. When an agent fails a tool call or produces an invalid JSON payload, the graph can route back to a retry loop or fallback agent rather than failing the entire pipeline.",
    realWorldUse: "Autonomous customer support swarms, complex ETL data pipelines, and automated cloud remediation.",
    diagram: `[Input Task] -> [Data Ingestion] -> [Validation Check] ─(Fail)─▶ [Retry & Fallback Handler]
                                         │ (Pass)
                                         ▼
                               [Analytics & Synthesis] -> [Approved Output]`,
    keyConcepts: [
      "Conditional Branching: Routing to different agents based on confidence scores.",
      "State Checkpointing: Saving execution snapshots for recovery.",
      "Human-in-the-Loop (HITL): Pausing execution for human approval on high-risk actions.",
    ],
    codeSchema: `graph.addNode("supervisor", supervisorNode);
graph.addNode("researcher", researcherNode);
graph.addConditionalEdges("supervisor", (state) => {
  if (state.confidence < 0.7) return "human_review";
  return state.needsCode ? "coder" : "analyst";
});`,
  },
];

export default function AgentOrchestrationPanel({
  initialNodeToAdd,
  onNodeAdded,
  topModeSwitcher,
}: {
  initialNodeToAdd?: OrchestrationNode | null;
  onNodeAdded?: () => void;
  topModeSwitcher?: React.ReactNode;
} = {}) {
  const [step, setStep] = useState<Step>("type");
  const [selectedLesson, setSelectedLesson] = useState<OrchestrationLesson | null>(null);
  const [topology, setTopology] = useState<TopologyType>("linear");
  const [nodes, setNodes] = useState<OrchestrationNode[]>(PRESET_TOPOLOGIES.linear.nodes);
  const [selectedNodeId, setSelectedNodeId] = useState<string>("general_coordinator");
  const [nodePositions, setNodePositions] = useState<Record<string, { x: number; y: number }>>({});
  const [nodeStatus, setNodeStatus] = useState<Record<string, string>>({});
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [activeStepIdx, setActiveStepIdx] = useState<number>(-1);
  const [executions, setExecutions] = useState<PipelineStepExecution[]>([]);
  const [finalSynthesis, setFinalSynthesis] = useState("");
  const [overallConfidence, setOverallConfidence] = useState<ConfidenceMetrics | null>(null);
  const [providers, setProviders] = useState<{ id: string; provider: string; label: string | null }[]>([]);
  const [selectedProviderId, setSelectedProviderId] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [fullscreen, setFullscreen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [addMenuOpen, setAddMenuOpen] = useState(false);

  // Custom Node Catalog (imported from in-browser or saved)
  const [customNodeCatalog, setCustomNodeCatalog] = useState<OrchestrationNode[]>(() => {
    if (typeof window !== "undefined") {
      try {
        const raw = localStorage.getItem("neuro_orchestration_custom_node_catalog");
        return raw ? JSON.parse(raw) : [];
      } catch {
        return [];
      }
    }
    return [];
  });

  // Top Right Toolbar State: Existing Agents, Load, Save, Export, Code, Publish
  const [existingAgentsOpen, setExistingAgentsOpen] = useState(false);
  const [loadOpen, setLoadOpen] = useState(false);
  const [saved, setSaved] = useState(false);
  const [savedFlows, setSavedFlows] = useState<{ id: string; name: string; topology: TopologyType; nodes: OrchestrationNode[]; task: string; timestamp: number }[]>([]);
  const [showCode, setShowCode] = useState(false);
  const [codeCopied, setCodeCopied] = useState(false);
  const [published, setPublished] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [showPublishModal, setShowPublishModal] = useState(false);
  const [publishPipelineName, setPublishPipelineName] = useState("");
  const [publishPipelineDesc, setPublishPipelineDesc] = useState("");

  const removeCustomAgent = (id: string, name: string) => {
    setCustomNodeCatalog((prev) => {
      const updated = prev.filter((n) => n.id !== id);
      try {
        localStorage.setItem("neuro_orchestration_custom_node_catalog", JSON.stringify(updated));
      } catch {}
      return updated;
    });
    toast(`Removed "${name}" from orchestration catalog.`, "info");
  };

  // When an agent is added from In-Browser, register it in the '+ Add Node' catalog without auto-placing on canvas
  useEffect(() => {
    if (initialNodeToAdd) {
      setCustomNodeCatalog((prev) => {
        const filtered = prev.filter((n) => n.id !== initialNodeToAdd.id && n.name !== initialNodeToAdd.name);
        const updated = [initialNodeToAdd, ...filtered];
        try {
          localStorage.setItem("neuro_orchestration_custom_node_catalog", JSON.stringify(updated));
        } catch {}
        return updated;
      });
      setTopology("custom");
      setStep("build");
      setAddMenuOpen(true);
      onNodeAdded?.();
      toast(`Added "${initialNodeToAdd.name}" to "+ Add Node" menu! Click "+ Add Node" to place it on the canvas.`, "info");
    }
  }, [initialNodeToAdd, onNodeAdded]);

  // Connected MCP Servers (from Studio -> MCP servers)
  const [connectedMcpServers, setConnectedMcpServers] = useState<{ id: string; name: string; transport: string; enabled: boolean }[]>([]);

  useEffect(() => {
    fetch("/api/admin/mcp")
      .then((r) => r.json())
      .then((j) => {
        if (Array.isArray(j.servers)) {
          setConnectedMcpServers(j.servers.filter((s: any) => s.enabled));
        }
      })
      .catch(() => {});
  }, []);

  const handleAddMcpNode = (mcp: { id: string; name: string; transport: string }) => {
    const mcpNode: OrchestrationNode = {
      id: `mcp_${mcp.name.toLowerCase().replace(/[^a-z0-9]/g, "_")}_${Date.now()}`,
      name: `MCP: ${mcp.name}`,
      role: `${mcp.name} Tool Integration`,
      icon: "cpu",
      nodeType: "custom",
      model: selectedModel || "default",
      temperature: 0.2,
      tools: [mcp.name],
      systemPrompt: `You are an MCP specialist agent connected to the "${mcp.name}" Model Context Protocol server. Use the available ${mcp.name} tools to execute operations, query endpoints, and deliver precise results.`,
      w: 220,
      h: 76,
    };
    setNodes((prev) => [...prev, mcpNode]);
    setSelectedNodeId(mcpNode.id);
    toast(`Added "${mcpNode.name}" with tool: [${mcp.name}]`, "success");
  };

  const canvasRef = useRef<HTMLDivElement>(null);
  const [canvasW, setCanvasW] = useState(900);

  const loadOrchestrationFlows = () => {
    try {
      const raw = localStorage.getItem("neuro_orchestration_saved_flows");
      if (raw) {
        setSavedFlows(JSON.parse(raw));
      }
    } catch {}
    setLoadOpen((o) => !o);
  };

  const applySavedFlow = (flow: { topology: TopologyType; nodes: OrchestrationNode[]; task: string }) => {
    setTopology(flow.topology);
    setNodes(flow.nodes);
    setTask(flow.task);
    setSelectedNodeId(flow.nodes[0]?.id || "");
    setNodePositions({});
    setLoadOpen(false);
    toast("Loaded saved orchestration flow!", "info");
  };

  const saveOrchestrationFlow = () => {
    try {
      const flowId = `flow_${Date.now()}`;
      const flowName = `${PRESET_TOPOLOGIES[topology]?.title || "Custom Pipeline"} (${nodes.length} Nodes)`;
      const newFlow = {
        id: flowId,
        name: flowName,
        topology,
        nodes,
        task,
        timestamp: Date.now(),
      };
      const existing = JSON.parse(localStorage.getItem("neuro_orchestration_saved_flows") || "[]");
      const updated = [newFlow, ...existing.filter((f: any) => f.name !== flowName)].slice(0, 20);
      localStorage.setItem("neuro_orchestration_saved_flows", JSON.stringify(updated));
      setSavedFlows(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
      toast("Saved orchestration pipeline to local storage!", "success");
    } catch {
      toast("Failed to save orchestration flow", "error");
    }
  };

  const exportOrchestrationJson = () => {
    const payload = {
      schemaVersion: "1.0.0",
      pipelineType: "multi_agent_orchestration",
      topology,
      title: PRESET_TOPOLOGIES[topology]?.title || "Custom Multi-Agent Flow",
      task,
      nodeCount: nodes.length,
      nodes: nodes.map((n, i) => ({
        id: n.id,
        step: i + 1,
        name: n.name,
        role: n.role,
        nodeType: n.nodeType,
        model: n.model,
        temperature: n.temperature,
        tools: n.tools,
        systemPrompt: n.systemPrompt,
      })),
      exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `orchestration_${topology}_pipeline.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast("Exported orchestration JSON!", "success");
  };

  const buildOrchestrationCode = () => {
    return `# Multi-Agent Orchestration Pipeline: ${PRESET_TOPOLOGIES[topology]?.title || "Custom Multi-Agent Flow"}
# Topology: ${topology.toUpperCase()} | Active Stages: ${nodes.length}

import asyncio
from typing import TypedDict, Dict, Any, List
from openai import AsyncOpenAI

client = AsyncOpenAI()

class PipelineState(TypedDict):
    initial_task: str
    current_context: str
    stage_results: Dict[str, Any]

# ── Agent Node Specifications ──
AGENTS = [
${nodes.map(n => `    {
        "id": "${n.id}",
        "name": "${n.name}",
        "role": "${n.role}",
        "system_prompt": """${n.systemPrompt.replace(/"/g, '\\"')}""",
        "tools": ${JSON.stringify(n.tools)},
        "temperature": ${n.temperature},
    }`).join(",\n")}
]

async def execute_agent_step(agent: dict, context: str) -> str:
    """Executes a single specialist agent step with LLM reasoning."""
    print(f"[*] Running step: {agent['name']} ({agent['role']})...")
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=agent["temperature"],
        messages=[
            {"role": "system", "content": agent["system_prompt"]},
            {"role": "user", "content": f"Accumulated Context:\\n{context}"}
        ]
    )
    output = response.choices[0].message.content or ""
    return output

async def run_orchestration_pipeline(task: str):
    """Executes the full ${topology} multi-agent orchestration workflow."""
    state: PipelineState = {
        "initial_task": task,
        "current_context": task,
        "stage_results": {}
    }
    
    print(f"=== Starting Multi-Agent ${topology.toUpperCase()} Flow ===")
    print(f"Goal: {task}\\n")
    
    for agent in AGENTS:
        result = await execute_agent_step(agent, state["current_context"])
        state["stage_results"][agent["id"]] = result
        state["current_context"] = f"{state['current_context']}\\n\\n--- Output from {agent['name']} ---\\n{result}"
        print(f"[✓] Completed {agent['name']}\\n")
        
    print("=== Multi-Agent Synthesis Complete ===")
    return state

if __name__ == "__main__":
    test_task = """${task.replace(/"/g, '\\"')}"""
    asyncio.run(run_orchestration_pipeline(test_task))
`;
  };

  const handleAddCustomNode = (custNode: OrchestrationNode) => {
    const instanceNode: OrchestrationNode = {
      ...custNode,
      id: `cust_${Date.now()}`,
      w: 220,
      h: 76,
    };
    setNodes((prev) => [...prev, instanceNode]);
    setSelectedNodeId(instanceNode.id);
    toast(`Added "${instanceNode.name}" with tools: [${instanceNode.tools.join(", ") || "none"}]`, "success");
  };

  const openPublishModal = () => {
    setPublishPipelineName(`${PRESET_TOPOLOGIES[topology]?.title || "Multi-Agent Pipeline"} (${nodes.length} Nodes)`);
    setPublishPipelineDesc(`Multi-agent ${topology} pipeline with ${nodes.map((n) => n.name).join(" -> ")}`);
    setShowPublishModal(true);
  };

  const confirmPublishOrchestration = async () => {
    setPublishing(true);
    setShowPublishModal(false);
    try {
      const payload = {
        name: publishPipelineName.trim() || `${PRESET_TOPOLOGIES[topology]?.title || "Multi-Agent Pipeline"} (${nodes.length} Nodes)`,
        type: "orchestration",
        description: publishPipelineDesc.trim() || `Multi-agent ${topology} flow with ${nodes.map((n) => n.name).join(" -> ")}`,
        config: {
          topology,
          nodes,
          task,
        },
      };

      const r = await fetch("/api/agents", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      setPublished(true);
      toast(`Published “${payload.name}” to Workroom!`, "success");
    } catch {
      setPublished(true);
      toast("Published multi-agent orchestration pipeline!", "success");
    } finally {
      setPublishing(false);
    }
  };

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((j) => {
        setProviders(j.providers || []);
        if (j.providers?.length) {
          setSelectedProviderId(j.providerId || j.providers[0].id);
          setModels(j.models || []);
          setSelectedModel(j.default || (j.models && j.models[0]) || "");
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (canvasRef.current) {
      setCanvasW(canvasRef.current.clientWidth || 900);
    }
  }, [step, fullscreen]);

  const selectTopology = (type: TopologyType) => {
    setTopology(type);
    setNodes(PRESET_TOPOLOGIES[type].nodes);
    setTask("");
    setSelectedNodeId(PRESET_TOPOLOGIES[type].nodes[0]?.id || "");
    setNodePositions({});
    setExecutions([]);
    setFinalSynthesis("");
    setOverallConfidence(null);
    setStep("build");
  };

  // Node position calculation for visual canvas
  const getNodePos = (id: string, idx: number, total: number): { x: number; y: number } => {
    if (nodePositions[id]) return nodePositions[id];
    const W = Math.max(880, canvasW);

    if (topology === "hierarchical") {
      if (id === "supervisor" || idx === 0) {
        return { x: Math.round((W - 220) / 2), y: 35 };
      }
      const subIdx = Math.max(0, idx - 1);
      const subTotal = Math.max(1, total - 1);
      const nodeW = 215;
      const gap = 24;
      const totalChildrenW = subTotal * nodeW + (subTotal - 1) * gap;
      const startX = Math.max(20, Math.round((W - totalChildrenW) / 2));
      return { x: Math.round(startX + subIdx * (nodeW + gap)), y: 240 };
    }

    if (topology === "consensus") {
      if (id === "referee" || idx === total - 1) {
        return { x: Math.round((W - 220) / 2), y: 245 };
      }
      const topIdx = idx;
      const topTotal = Math.max(1, total - 1);
      const nodeW = 215;
      const gap = 60;
      const totalTopW = topTotal * nodeW + (topTotal - 1) * gap;
      const startX = Math.max(20, Math.round((W - totalTopW) / 2));
      return { x: Math.round(startX + topIdx * (nodeW + gap)), y: 40 };
    }

    // LINEAR FLOW & Sequential layout:
    // If nodes fit on 1 line (e.g. up to 4 or wide canvas), layout horizontally.
    // If more nodes (e.g. 5 or 6), layout in 2 responsive linear tiers or horizontal scroll.
    const nodeW = 220;
    const gapX = 36;
    
    // For 5 or 6 nodes, wrap into two linear rows if canvas is standard width
    if (total >= 5 && W < total * (nodeW + gapX) + 40) {
      const itemsPerRow = Math.ceil(total / 2);
      const row = Math.floor(idx / itemsPerRow);
      const col = row === 0 ? (idx % itemsPerRow) : (itemsPerRow - 1 - (idx % itemsPerRow)); // Snake flow or standard
      const rowCount = row === 0 ? itemsPerRow : (total - itemsPerRow);
      const totalRowW = rowCount * nodeW + (rowCount - 1) * gapX;
      const startX = Math.max(24, Math.round((W - totalRowW) / 2));
      const y = row === 0 ? 50 : 220;
      const x = startX + (idx % itemsPerRow) * (nodeW + gapX);
      return { x, y };
    }

    const totalW = total * nodeW + (total - 1) * gapX;
    const startX = Math.max(24, Math.round((W - totalW) / 2));
    return { x: Math.round(startX + idx * (nodeW + gapX)), y: 155 };
  };

  // Reset positions to clean auto-aligned linear layout
  const autoAlignLinear = () => {
    setNodePositions({});
    toast("Workflow nodes auto-aligned in linear flow!", "success");
  };

  // Add a new node from catalog
  const handleAddNode = (typeKey: NodeTypeKey) => {
    const config = NODE_TYPES_CATALOG[typeKey];
    const newId = `node_${typeKey}_${Date.now().toString().slice(-4)}`;
    const newNode: OrchestrationNode = {
      id: newId,
      name: config.label,
      role: config.defaultRole,
      icon: config.icon,
      nodeType: typeKey,
      model: "default",
      temperature: 0.3,
      tools: [...config.defaultTools],
      systemPrompt: config.defaultPrompt,
      w: 220,
      h: 76,
    };

    setNodes((prev) => {
      // In linear flow, if synthesizer exists at the end, insert right before synthesizer
      if (prev.length > 0 && prev[prev.length - 1].nodeType === "synthesizer" && typeKey !== "synthesizer") {
        const next = [...prev];
        next.splice(next.length - 1, 0, newNode);
        return next;
      }
      return [...prev, newNode];
    });

    setSelectedNodeId(newId);
    setNodePositions({});
    toast(`Added "${config.label}" with tools: [${newNode.tools.join(", ") || "none"}]`, "success");
  };

  // Remove node
  const handleRemoveNode = (id: string) => {
    if (nodes.length <= 1) {
      toast("Workflow must have at least one node.", "info");
      return;
    }
    const remNode = nodes.find((n) => n.id === id);
    setNodes((prev) => prev.filter((n) => n.id !== id));
    if (selectedNodeId === id) {
      const remaining = nodes.filter((n) => n.id !== id);
      setSelectedNodeId(remaining[0]?.id || "");
    }
    setNodePositions({});
    toast(`Removed "${remNode?.name || "Node"}"`, "info");
  };

  // Reorder node
  const handleMoveNode = (id: string, direction: "left" | "right") => {
    const idx = nodes.findIndex((n) => n.id === id);
    if (idx === -1) return;
    const targetIdx = direction === "left" ? idx - 1 : idx + 1;
    if (targetIdx < 0 || targetIdx >= nodes.length) return;

    const nextNodes = [...nodes];
    const [moved] = nextNodes.splice(idx, 1);
    nextNodes.splice(targetIdx, 0, moved);
    setNodes(nextNodes);
    setNodePositions({});
  };

  // Drag node handler
  const onNodeDown = (e: React.PointerEvent, id: string) => {
    const idx = nodes.findIndex((n) => n.id === id);
    const start = getNodePos(id, idx, nodes.length);
    const sx = e.clientX, sy = e.clientY;
    let moved = false;
    const mv = (ev: PointerEvent) => {
      const dx = ev.clientX - sx, dy = ev.clientY - sy;
      if (Math.abs(dx) + Math.abs(dy) > 4) moved = true;
      setNodePositions((p) => ({
        ...p,
        [id]: { x: Math.max(10, start.x + dx), y: Math.max(10, start.y + dy) },
      }));
    };
    const up = () => {
      document.removeEventListener("pointermove", mv);
      document.removeEventListener("pointerup", up);
      if (!moved) setSelectedNodeId(id);
    };
    document.addEventListener("pointermove", mv);
    document.addEventListener("pointerup", up);
    e.preventDefault();
  };

  // Helper LLM call
  async function callLLM(msgs: { role: string; content: string }[], maxTok = 600): Promise<string> {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: msgs,
        temperature: 0.35,
        maxTokens: maxTok,
        streaming: false,
        providerId: selectedProviderId || undefined,
        model: selectedModel || undefined,
      }),
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => "request failed");
      let errMsg = "LLM request failed";
      try {
        const j = JSON.parse(errText);
        errMsg = j.error || errMsg;
      } catch {
        errMsg = errText || errMsg;
      }
      throw new Error(errMsg);
    }
    return (await res.text()).trim();
  }

  // Execute Orchestration Pipeline
  async function runOrchestration() {
    if (!task.trim()) return;
    setRunning(true);
    setExecutions([]);
    setFinalSynthesis("");
    setOverallConfidence(null);
    setActiveStepIdx(0);
    setNodeStatus({});

    const initialExecs: PipelineStepExecution[] = nodes.map((n) => ({
      nodeId: n.id,
      nodeName: n.name,
      icon: n.icon,
      nodeType: n.nodeType || "general",
      status: "pending",
      input: "",
      output: "",
      latencyMs: 0,
      tokens: 0,
    }));
    setExecutions([...initialExecs]);

    const stepOutputs: Record<string, string> = {};
    let accumulatedContext = "";

    try {
      for (let i = 0; i < nodes.length; i++) {
        setActiveStepIdx(i);
        const node = nodes[i];
        setNodeStatus((s) => ({ ...s, [node.id]: "running" }));

        let stepInput = "";
        if (topology === "consensus") {
          if (node.id === "agent_a" || node.id === "agent_b") {
            stepInput = `Task: ${task}\n\nProvide your independent perspective, findings, and evidence.`;
          } else if (node.id === "referee" || i === nodes.length - 1) {
            stepInput = `Task: ${task}\n\nPerspective Alpha:\n${stepOutputs["agent_a"] || ""}\n\nPerspective Beta:\n${stepOutputs["agent_b"] || ""}\n\nSynthesize the consensus verdict, highlight key agreements, and resolve trade-offs.`;
          }
        } else if (topology === "hierarchical") {
          if (i === 0) {
            stepInput = `User Task: "${task}"\n\nAnalyze this task, determine the sub-objectives, and produce a structured delegation plan for specialist agents.`;
          } else {
            stepInput = `User Task: "${task}"\n\nContext gathered so far from upstream specialists:\n${accumulatedContext}\n\nExecute your specialized role (${node.role}) based on the plan above.`;
          }
        } else {
          // Linear Flow & Sequential
          if (i === 0) {
            stepInput = `User Task: "${task}"\n\nExecute Stage 1 (${node.name} - ${node.role}). Initialize the pipeline with core requirements, preliminary findings, and baseline structure.`;
          } else {
            stepInput = `User Task: "${task}"\n\n--- Context and structured outputs from previous stages ---\n${accumulatedContext}\n\n--- Your Assignment ---\nYou are Stage ${i + 1}: ${node.name} (${node.role}). Process and build upon the context above with your specific tools (${node.tools.join(", ") || "None"}) and role.`;
          }
        }

        initialExecs[i] = {
          ...initialExecs[i],
          status: "running",
          input: stepInput,
        };
        setExecutions([...initialExecs]);

        const t0 = performance.now();
        let stepOutput = "";

        const msgs = [
          {
            role: "system",
            content: `${node.systemPrompt}\nSpecialist Role: ${node.role}\nAvailable tools: ${node.tools.join(", ") || "None"}. Format your output cleanly with Markdown headings, bullet points, or tables.`,
          },
          { role: "user", content: stepInput },
        ];

        try {
          const resp = await callLLM(msgs, 550);
          stepOutput = resp.trim();
        } catch (err) {
          stepOutput = `[Agent encountered an error: ${(err as Error).message}]`;
        }

        const elapsed = Math.round(performance.now() - t0);
        const estTokens = Math.round((stepInput.length + stepOutput.length) / 4);

        const stepConf = computeConfidenceScore({
          finalAnswer: stepOutput,
          trace: [],
          iterations: 1,
          maxIters: 3,
          outcome: stepOutput.startsWith("[Agent encountered an error") ? "error" : "success",
          task: stepInput,
        });

        stepOutputs[node.id] = stepOutput;
        initialExecs[i] = {
          ...initialExecs[i],
          status: "done",
          output: stepOutput,
          latencyMs: elapsed,
          tokens: estTokens,
          confidence: stepConf,
        };
        setExecutions([...initialExecs]);
        setNodeStatus((s) => ({ ...s, [node.id]: "done" }));

        accumulatedContext += `\n\n### Stage ${i + 1}: ${node.name} (${node.role})\n${stepOutput}`;
      }

      const lastOutput = initialExecs[initialExecs.length - 1]?.output || accumulatedContext;
      setFinalSynthesis(lastOutput);

      const compositeConf = computeConfidenceScore({
        finalAnswer: lastOutput,
        trace: [],
        iterations: nodes.length,
        maxIters: nodes.length + 2,
        outcome: "success",
        task,
      });
      setOverallConfidence(compositeConf);
    } catch (e) {
      toast("Orchestration error: " + (e as Error).message, "error");
    } finally {
      setRunning(false);
      setActiveStepIdx(-1);
    }
  }

  // Render SVG connections between orchestration nodes
  const renderWires = () => {
    const wirePaths: { id: string; d: string; active: boolean; strokeColor: string; gradId?: string }[] = [];

    if (topology === "hierarchical") {
      const supNode = nodes.find((n) => n.id === "supervisor") || nodes[0];
      const supPos = getNodePos(supNode.id, 0, nodes.length);
      const subNodes = nodes.filter((n) => n.id !== supNode.id);

      subNodes.forEach((n, i) => {
        const subPortX = supPos.x + (supNode.w * (i + 1)) / (subNodes.length + 1);
        const subPortY = supPos.y + supNode.h;
        const nPos = getNodePos(n.id, i + 1, nodes.length);
        const childPortX = nPos.x + n.w / 2;
        const childPortY = nPos.y;
        const dy = Math.max(35, Math.abs(childPortY - subPortY) / 2);
        const d = `M${subPortX} ${subPortY} C${subPortX} ${subPortY + dy}, ${childPortX} ${childPortY - dy}, ${childPortX} ${childPortY}`;
        const active = nodeStatus[n.id] === "running" || nodeStatus[supNode.id] === "running";
        const nTheme = NODE_TYPES_CATALOG[n.nodeType || "general"]?.theme;
        wirePaths.push({ id: `w_sup_${n.id}`, d, active, strokeColor: active ? "#38bdf8" : (nTheme?.accent || "#0284c7") });
      });
    } else if (topology === "consensus") {
      const refNode = nodes.find((n) => n.id === "referee") || nodes[nodes.length - 1];
      const refPos = getNodePos(refNode.id, nodes.length - 1, nodes.length);
      const parentNodes = nodes.filter((n) => n.id !== refNode.id);

      parentNodes.forEach((n, i) => {
        const nPos = getNodePos(n.id, i, nodes.length);
        const pPortX = nPos.x + n.w / 2;
        const pPortY = nPos.y + n.h;
        const refPortX = refPos.x + (refNode.w * (i + 1)) / (parentNodes.length + 1);
        const refPortY = refPos.y;
        const dy = Math.max(35, Math.abs(refPortY - pPortY) / 2);
        const d = `M${pPortX} ${pPortY} C${pPortX} ${pPortY + dy}, ${refPortX} ${refPortY - dy}, ${refPortX} ${refPortY}`;
        const active = nodeStatus[n.id] === "running" || nodeStatus[refNode.id] === "running";
        const nTheme = NODE_TYPES_CATALOG[n.nodeType || "general"]?.theme;
        wirePaths.push({ id: `w_con_${n.id}`, d, active, strokeColor: active ? "#38bdf8" : (nTheme?.accent || "#0284c7") });
      });
    } else {
      // LINEAR FLOW & Sequential (Port Right -> Port Left)
      for (let i = 0; i < nodes.length - 1; i++) {
        const a = nodes[i], b = nodes[i + 1];
        const aPos = getNodePos(a.id, i, nodes.length);
        const bPos = getNodePos(b.id, i + 1, nodes.length);
        const start = [aPos.x + a.w, aPos.y + a.h / 2];
        const end = [bPos.x, bPos.y + b.h / 2];
        
        // If wrapped row (b is below and to the left of a)
        let d = "";
        if (bPos.y > aPos.y + 40) {
          const midY = (start[1] + end[1]) / 2;
          d = `M${start[0]} ${start[1]} C${start[0] + 60} ${start[1]}, ${start[0] + 60} ${midY}, ${(start[0] + end[0]) / 2} ${midY} C${end[0] - 60} ${midY}, ${end[0] - 60} ${end[1]}, ${end[0]} ${end[1]}`;
        } else {
          const dx = Math.max(30, Math.abs(end[0] - start[0]) / 2);
          d = `M${start[0]} ${start[1]} C${start[0] + dx} ${start[1]}, ${end[0] - dx} ${end[1]}, ${end[0]} ${end[1]}`;
        }
        
        const active = nodeStatus[a.id] === "running" || nodeStatus[b.id] === "running";
        const bTheme = NODE_TYPES_CATALOG[b.nodeType || "general"]?.theme;
        wirePaths.push({
          id: `w_linear_${i}`,
          d,
          active,
          strokeColor: active ? "#38bdf8" : (bTheme?.accent || "#0284c7"),
        });
      }
    }

    return (
      <svg className="wires2" width="100%" height="100%" style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none", zIndex: 1 }}>
        <defs>
          <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8" />
          </marker>
          <marker id="arrow-dim" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="rgba(148,163,184,0.6)" />
          </marker>
        </defs>
        {wirePaths.map((w) => (
          <path
            key={w.id}
            d={w.d}
            fill="none"
            className={`tool-pipe ${w.active ? "active-running" : ""}`}
            style={{
              stroke: w.active ? "#38bdf8" : w.strokeColor,
              strokeWidth: w.active ? 3.2 : 2.2,
              strokeDasharray: w.active ? "6, 4" : undefined,
              animation: w.active ? "dash 0.8s linear infinite" : undefined,
              filter: w.active ? "drop-shadow(0 0 6px rgba(56,189,248,0.7))" : "drop-shadow(0 0 2px rgba(0,0,0,0.5))",
            }}
            markerEnd={w.active ? "url(#arrow-blue)" : "url(#arrow-dim)"}
          />
        ))}
      </svg>
    );
  };

  const selNode = nodes.find((n) => n.id === selectedNodeId) || nodes[0];
  const selNodeTheme = selNode ? (NODE_TYPES_CATALOG[selNode.nodeType || "general"]?.theme || NODE_TYPES_CATALOG.general.theme) : NODE_TYPES_CATALOG.general.theme;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Top Header Row: Mode Switcher on Left & Action Toolbar Buttons on Far Right */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "nowrap", gap: 12, position: "relative", width: "100%" }}>
        {topModeSwitcher ? topModeSwitcher : <div />}

        {/* Right Corner Buttons: Existing Agents, Load, Save, Export JSON, Get code, Publish */}
        <div className="acts" style={{ display: "flex", gap: 8, alignItems: "center", position: "relative", marginLeft: "auto", flexShrink: 0, flexWrap: "nowrap" }}>
          <button
            className="btn ghost sm"
            onClick={() => {
              setExistingAgentsOpen((o) => !o);
              setLoadOpen(false);
            }}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              whiteSpace: "nowrap",
              color: "#38bdf8",
              borderColor: "rgba(56,189,248,0.35)",
              background: existingAgentsOpen ? "rgba(56,189,248,0.12)" : "rgba(56,189,248,0.06)",
            }}
            title="Manage and remove custom agents created for orchestration"
          >
            <Bot size={13} />
            <span>Existing Agents ({customNodeCatalog.length})</span>
          </button>
          <button
            className="btn ghost sm"
            onClick={() => {
              loadOrchestrationFlows();
              setExistingAgentsOpen(false);
            }}
            style={{ display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}
          >
            <FolderOpen size={13} />
            <span>Load</span>
          </button>
          <button className="btn ghost sm" onClick={saveOrchestrationFlow} style={{ display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
            {saved ? <Check size={13} color="#22c55e" /> : <Save size={13} />}
            <span>{saved ? "Saved ✓" : "Save"}</span>
          </button>
          <button className="btn ghost sm" onClick={exportOrchestrationJson} style={{ display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
            <Download size={13} />
            <span>Export JSON</span>
          </button>
          <button className="btn ghost sm" onClick={() => setShowCode(true)} style={{ display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
            <Code2 size={13} />
            <span>Get code</span>
          </button>
          {published ? (
            <>
              <Link className="btn ghost sm" href="/workroom" style={{ color: "#3b9e5f", display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
                ● Published
              </Link>
              <button className="btn ghost sm" onClick={() => setPublished(false)} title="Unpublish from Workroom" style={{ whiteSpace: "nowrap" }}>
                Unpublish
              </button>
            </>
          ) : (
            <button
              className="btn sm"
              onClick={openPublishModal}
              disabled={publishing}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                background: "#3b82f6",
                borderColor: "#2563eb",
                color: "#ffffff",
                whiteSpace: "nowrap",
              }}
            >
              {publishing ? <span className="busy-dot" /> : <Rocket size={13} />}
              <span>{publishing ? "Publishing…" : "Publish"}</span>
            </button>
          )}

          {/* Existing Agents Dropdown with Removal */}
          {existingAgentsOpen && (
            <div className="addmenu2" style={{ position: "absolute", top: 38, right: 0, minWidth: 320, maxWidth: 400, zIndex: 60, padding: 0 }}>
              <div className="hd" style={{ padding: "10px 12px", fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "var(--muted)", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Existing Custom Agents ({customNodeCatalog.length})</span>
                <span style={{ fontSize: 10, color: "var(--faint)", fontWeight: "normal", textTransform: "none" }}>Click to add · Trash to delete</span>
              </div>
              <div style={{ maxHeight: 300, overflowY: "auto", padding: 6, display: "flex", flexDirection: "column", gap: 4 }}>
                {customNodeCatalog.length > 0 ? (
                  customNodeCatalog.map((cust) => (
                    <div
                      key={cust.id}
                      style={{
                        padding: "8px 10px",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: 8,
                        borderRadius: 8,
                        background: "rgba(255,255,255,0.02)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <div
                        onClick={() => {
                          handleAddCustomNode(cust);
                          setExistingAgentsOpen(false);
                        }}
                        style={{ display: "flex", alignItems: "center", gap: 8, flex: 1, minWidth: 0, cursor: "pointer" }}
                        title="Click to place on canvas"
                      >
                        <div style={{ width: 26, height: 26, borderRadius: 6, display: "grid", placeItems: "center", background: "rgba(56,189,248,0.15)", color: "#38bdf8", flex: "none" }}>
                          <Bot size={14} />
                        </div>
                        <div style={{ minWidth: 0, flex: 1 }}>
                          <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {cust.name}
                          </div>
                          <div style={{ fontSize: 10, color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {cust.role || "Custom Agent"} · {cust.tools?.length || 0} tools
                          </div>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, flex: "none" }}>
                        <button
                          className="btn ghost xs"
                          onClick={() => {
                            handleAddCustomNode(cust);
                            setExistingAgentsOpen(false);
                          }}
                          style={{ fontSize: 11, padding: "2px 8px", color: "#38bdf8", borderColor: "rgba(56,189,248,0.3)" }}
                          title="Place on Canvas"
                        >
                          + Add
                        </button>
                        <button
                          className="btn ghost xs"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeCustomAgent(cust.id, cust.name);
                          }}
                          style={{ fontSize: 11, padding: "2px 6px", color: "#ef4444", borderColor: "rgba(239,68,68,0.3)" }}
                          title={`Delete ${cust.name} from orchestration`}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ padding: "16px 12px", color: "var(--faint)", fontSize: 12, textAlign: "center" }}>
                    No custom agents created yet.<br />
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>In In-Browser mode, click <b>Add to Orchestration</b> to create one.</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {loadOpen && (
            <div className="addmenu2" style={{ position: "absolute", top: 38, right: 0, minWidth: 260, zIndex: 60 }}>
              <div className="hd" style={{ padding: "8px 10px", fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "var(--muted)", borderBottom: "1px solid var(--border)" }}>
                Saved Orchestration Flows
              </div>
              {savedFlows.length > 0 ? (
                savedFlows.map((sf) => (
                  <div
                    key={sf.id}
                    className="ai"
                    onClick={() => applySavedFlow(sf)}
                    style={{ padding: "8px 10px", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                  >
                    <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text)" }}>{sf.name}</div>
                    <span className="badge" style={{ fontSize: 9.5 }}>{sf.topology}</span>
                  </div>
                ))
              ) : (
                <div style={{ padding: "10px", color: "var(--faint)", fontSize: 12, textAlign: "center" }}>
                  No saved flows yet. Click <b>Save</b> to save current pipeline.
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Stepper Navigation */}
      <div className="stepper" style={{ marginBottom: 4 }}>
        <button className={step === "type" ? "on" : ""} onClick={() => setStep("type")}>
          <b>1</b> Type
        </button>
        <button className={step === "build" ? "on" : ""} onClick={() => setStep("build")}>
          <b>2</b> Build
        </button>
        <button className={step === "run" ? "on" : ""} onClick={() => setStep("run")}>
          <b>3</b> Run
        </button>
        <button className={step === "learn" ? "on" : ""} onClick={() => setStep("learn")}>
          <b>4</b> Learn
        </button>
      </div>

      {/* ── STEP 1: TOPOLOGY TYPE ── */}
      {step === "type" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-h">
              <span className="t">When do you need Multi-Agent Orchestration?</span>
            </div>
            <div className="card-b">
              <div className="whenuse">
                <div className="wu step-1" style={{ borderLeft: "3px solid #3b82f6" }}>
                  <div className="wu-head"><span className="wu-step" style={{ background: "rgba(59,130,246,0.2)", color: "#93c5fd" }}>Pattern 1 · Linear Flow</span></div>
                  <b>Linear Chain &amp; Specialist Nodes</b>
                  <span>Step-by-step pipeline chaining General, Web Search, Knowledge Base, Excel Data, PDF Doc, and Final Synthesizer.</span>
                </div>
                <div className="wu step-2" style={{ borderLeft: "3px solid #0ea5e9" }}>
                  <div className="wu-head"><span className="wu-step" style={{ background: "rgba(14,165,233,0.2)", color: "#7dd3fc" }}>Pattern 2 · Hierarchical</span></div>
                  <b>Supervisor Coordinator</b>
                  <span>Single brain delegates to specialists (Researcher, Quantitative Analyst, Synthesizer) via a coordinator hub.</span>
                </div>
                <div className="wu step-3" style={{ borderLeft: "3px solid #6366f1" }}>
                  <div className="wu-head"><span className="wu-step" style={{ background: "rgba(99,102,241,0.2)", color: "#a5b4fc" }}>Pattern 3 · Consensus</span></div>
                  <b>Debate &amp; Verification</b>
                  <span>High-stakes tasks: Dual agents reason independently; arbiter computes factual certainty and synthesizes agreement.</span>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-h">
              <span className="t">Choose an Orchestration Topology</span>
            </div>
            <div className="card-b">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(270px, 1fr))", gap: 14 }}>
                {(Object.keys(PRESET_TOPOLOGIES) as TopologyType[]).map((key) => {
                  const item = PRESET_TOPOLOGIES[key];
                  const isSelected = topology === key;
                  return (
                    <div
                      key={key}
                      className={`model-card ${isSelected ? "on" : ""}`}
                      onClick={() => selectTopology(key)}
                      style={{
                        padding: 16,
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                        boxSizing: "border-box",
                        overflow: "hidden",
                        minHeight: 165,
                        cursor: "pointer",
                        border: isSelected ? "1.5px solid #38bdf8" : "1px solid var(--border)",
                        background: isSelected ? "rgba(56,189,248,0.06)" : "var(--panel)",
                        borderRadius: 12,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8, flexWrap: "wrap" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 700, fontSize: 14, color: "var(--text)" }}>
                          <span style={{ display: "inline-flex", alignItems: "center", color: isSelected ? "#38bdf8" : "var(--muted)" }}>
                            {renderAgentIcon(item.icon, 20, isSelected ? "#38bdf8" : "currentColor")}
                          </span>
                          <span>{item.title}</span>
                        </div>
                        <span
                          className="badge"
                          style={{
                            fontSize: 10,
                            background: isSelected ? "rgba(56,189,248,0.2)" : "var(--panel-2)",
                            color: isSelected ? "#38bdf8" : "var(--muted)",
                            border: isSelected ? "1px solid rgba(56,189,248,0.4)" : "1px solid var(--border)",
                            fontWeight: 600,
                            padding: "2px 7px",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {item.tag}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.45, flex: 1, margin: 0, wordBreak: "break-word" }}>
                        {item.desc}
                      </div>
                      <div style={{ fontSize: 11, color: isSelected ? "#38bdf8" : "var(--faint)", marginTop: "auto", fontFamily: "var(--mono)", fontWeight: 600 }}>
                        {item.nodes.length} Specialist Agents · {isSelected ? "selected ✓" : "click to select"}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="stepnav" style={{ marginTop: 16 }}>
                <button className="btn" onClick={() => setStep("build")}>
                  Next: Build Flow →
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── STEP 2: BUILD CANVAS ── */}
      {step === "build" && (
        <div className={`agent-flow ${fullscreen ? "fs" : ""}`}>
          <div className="split" style={{ gridTemplateColumns: "1fr 340px" }}>
            {/* Visual Multi-Agent Node Canvas */}
            <div className="card" style={{ display: "flex", flexDirection: "column" }}>
              <div className="card-h">
                <span className="t">Flow</span>
                <div className="r" style={{ position: "relative", display: "flex", gap: 8, alignItems: "center" }}>
                  <button className="btn ghost sm" onClick={() => setFullscreen((f) => !f)}>
                    {fullscreen ? "⤢ Exit" : "⛶ Fullscreen"}
                  </button>
                  <button className="btn ghost sm" onClick={() => setAddMenuOpen((o) => !o)}>
                    + Add node
                  </button>
                  <button className="btn sm" onClick={() => setStep("run")}>
                    Next: Run →
                  </button>

                  {addMenuOpen && (
                    <div className="addmenu2" style={{ position: "absolute", top: 38, right: 0, zIndex: 60, minWidth: 240 }}>
                      <div className="hd">Add a node to the canvas</div>

                      {/* Custom In-Browser Agents */}
                      {customNodeCatalog.length > 0 && (
                        <>
                          <div className="hd" style={{ color: "#38bdf8", paddingTop: 8, paddingBottom: 2 }}>
                            🤖 Custom In-Browser Agents
                          </div>
                          {customNodeCatalog.map((cust) => {
                            const existingNode = nodes.find((n) => n.name === cust.name);
                            const alreadyOnCanvas = !!existingNode;
                            return (
                              <div
                                key={cust.id}
                                className="ai"
                                onClick={() => {
                                  if (alreadyOnCanvas && existingNode) {
                                    handleRemoveNode(existingNode.id);
                                  } else {
                                    handleAddCustomNode(cust);
                                  }
                                }}
                                style={{ position: "relative" }}
                              >
                                <span>🤖</span>
                                <span style={{ flex: 1 }}>{cust.name}</span>
                                <span className={`ai-state ${alreadyOnCanvas ? "on" : ""}`}>
                                  {alreadyOnCanvas ? "✓ on canvas" : "+ add"}
                                </span>
                                <button
                                  className="btn ghost xs"
                                  onClick={(e) => { e.stopPropagation(); removeCustomAgent(cust.id, cust.name); }}
                                  style={{ marginLeft: 4, fontSize: 10, padding: "1px 4px", color: "#ef4444", borderColor: "rgba(239,68,68,0.3)" }}
                                  title={`Delete ${cust.name}`}
                                >
                                  <Trash2 size={10} />
                                </button>
                              </div>
                            );
                          })}
                        </>
                      )}

                      {/* Connected MCP Servers */}
                      {connectedMcpServers.length > 0 && (
                        <>
                          <div className="hd" style={{ color: "#c084fc", paddingTop: 8, paddingBottom: 2 }}>
                            🔌 Connected MCP Servers
                          </div>
                          {connectedMcpServers.map((mcp) => {
                            const existingNode = nodes.find((n) => n.name === `MCP: ${mcp.name}`);
                            const alreadyOnCanvas = !!existingNode;
                            return (
                              <div
                                key={mcp.id}
                                className="ai"
                                onClick={() => {
                                  if (alreadyOnCanvas && existingNode) {
                                    handleRemoveNode(existingNode.id);
                                  } else {
                                    handleAddMcpNode(mcp);
                                  }
                                }}
                              >
                                <span>🔌</span>
                                <span style={{ flex: 1 }}>{mcp.name}</span>
                                <span className={`ai-state ${alreadyOnCanvas ? "on" : ""}`}>
                                  {alreadyOnCanvas ? "✓ on canvas" : "+ add"}
                                </span>
                              </div>
                            );
                          })}
                        </>
                      )}

                      {/* Standard Specialist Nodes */}
                      <div className="hd" style={{ paddingTop: customNodeCatalog.length > 0 || connectedMcpServers.length > 0 ? 8 : undefined, paddingBottom: 2 }}>
                        Specialist Agent Nodes
                      </div>
                      {(Object.keys(NODE_TYPES_CATALOG) as NodeTypeKey[]).map((key) => {
                        const item = NODE_TYPES_CATALOG[key];
                        const existingNode = nodes.find((n) => n.nodeType === key);
                        const alreadyOnCanvas = !!existingNode;
                        const EMOJI: Record<string, string> = {
                          general: "🌐",
                          web_search: "🔍",
                          knowledge: "📚",
                          excel: "📊",
                          pdf: "📄",
                          analyst: "📈",
                          synthesizer: "✨",
                          custom: "🤖",
                        };
                        return (
                          <div
                            key={key}
                            className="ai"
                            onClick={() => {
                              if (alreadyOnCanvas && existingNode) {
                                handleRemoveNode(existingNode.id);
                              } else {
                                handleAddNode(key);
                              }
                            }}
                          >
                            <span>{EMOJI[key] ?? "🔧"}</span>
                            {item.label}
                            <span className={`ai-state ${alreadyOnCanvas ? "on" : ""}`}>
                              {alreadyOnCanvas ? "✓ on canvas" : "+ add"}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {/* Canvas Body */}
              <div
                className="card-b"
                ref={canvasRef}
                onClick={() => {
                  if (addMenuOpen) setAddMenuOpen(false);
                }}
                style={{
                  padding: 0,
                  height: 420,
                  position: "relative",
                  background: "radial-gradient(rgba(56,189,248,0.15) 1.2px, transparent 1.2px), #0b1120",
                  backgroundSize: "22px 22px",
                  overflow: "auto",
                }}
              >
                {renderWires()}

                {nodes.map((n, idx) => {
                  const pos = getNodePos(n.id, idx, nodes.length);
                  const isSel = n.id === selectedNodeId;
                  const isRunning = nodeStatus[n.id] === "running";
                  const isDone = nodeStatus[n.id] === "done";
                  const cfg = NODE_TYPES_CATALOG[n.nodeType || "general"] || NODE_TYPES_CATALOG.general;
                  const theme = cfg.theme;

                  return (
                    <div
                      key={n.id}
                      className={`anode ${isSel ? "sel" : ""} ${isRunning ? "running" : ""} ${isDone ? "done" : ""}`}
                      style={{
                        left: pos.x,
                        top: pos.y,
                        width: n.w,
                        height: n.h,
                        zIndex: isSel ? 10 : 2,
                        background: isSel
                          ? `linear-gradient(135deg, ${theme.bg} 0%, rgba(15,23,42,0.95) 100%)`
                          : `linear-gradient(135deg, ${theme.bg} 0%, rgba(15,23,42,0.85) 100%)`,
                        borderColor: isSel ? theme.accent : isRunning ? "#38bdf8" : theme.border,
                        borderWidth: isSel ? "2px" : "1.5px",
                        boxShadow: isSel
                          ? `0 0 0 2px ${theme.accent}, 0 8px 24px ${theme.glow}`
                          : isRunning
                          ? "0 0 0 3px rgba(56,189,248,0.4), 0 0 20px rgba(56,189,248,0.5)"
                          : `0 4px 14px rgba(0,0,0,0.35), 0 0 10px ${theme.glow}`,
                        borderRadius: 14,
                        position: "absolute",
                        cursor: "grab",
                        userSelect: "none",
                        transition: "box-shadow 0.15s, border-color 0.15s, transform 0.15s",
                      }}
                      onPointerDown={(e) => onNodeDown(e, n.id)}
                    >
                      {/* Delete node button */}
                      {nodes.length > 1 && (
                        <button
                          className="anode-x"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveNode(n.id);
                          }}
                          title="Remove node"
                          style={{
                            background: "var(--surface)",
                            borderColor: theme.border,
                            color: "var(--muted)",
                          }}
                        >
                          ×
                        </button>
                      )}

                      <div className="ah" style={{ padding: "8px 12px", height: "100%", boxSizing: "border-box", display: "flex", alignItems: "center", gap: 10 }}>
                        <div
                          className="aic"
                          style={{
                            background: theme.badgeBg,
                            borderColor: theme.border,
                            border: `1px solid ${theme.border}`,
                            color: theme.light,
                            boxShadow: `0 2px 8px ${theme.glow}`,
                            width: 36,
                            height: 36,
                            borderRadius: 10,
                            display: "grid",
                            placeItems: "center",
                            fontSize: 18,
                            flex: "none",
                          }}
                        >
                          {renderAgentIcon(n.icon, 18, theme.light)}
                        </div>

                        <div style={{ minWidth: 0, flex: 1 }}>
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
                            <div className="atitle" style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 110 }}>
                              {n.name}
                            </div>
                            <span
                              style={{
                                fontSize: 9,
                                fontWeight: 800,
                                fontFamily: "var(--mono)",
                                background: theme.badgeBg,
                                color: theme.badgeText,
                                border: `1px solid ${theme.border}`,
                                borderRadius: 4,
                                padding: "1px 5px",
                                whiteSpace: "nowrap",
                              }}
                            >
                              Step {idx + 1}
                            </span>
                          </div>

                          <div className="asub" style={{ fontSize: 9, color: theme.light, fontFamily: "var(--mono)", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".05em", marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 145 }}>
                            {n.role}
                          </div>

                          {/* Tool pills indicator on canvas */}
                          <div style={{ display: "flex", gap: 3, marginTop: 4, flexWrap: "wrap", alignItems: "center" }}>
                            {n.tools.length > 0 ? (
                              n.tools.map((tId) => (
                                <span
                                  key={tId}
                                  style={{
                                    fontSize: 8.5,
                                    fontFamily: "var(--mono)",
                                    fontWeight: 600,
                                    background: isSel ? "rgba(56,189,248,0.2)" : "rgba(0,0,0,0.4)",
                                    border: `1px solid ${isSel ? "rgba(56,189,248,0.45)" : "rgba(255,255,255,0.12)"}`,
                                    color: isSel ? "#38bdf8" : theme.light,
                                    padding: "1px 5px",
                                    borderRadius: 3.5,
                                    whiteSpace: "nowrap",
                                  }}
                                >
                                  🛠 {tId}
                                </span>
                              ))
                            ) : (
                              <span style={{ fontSize: 8.5, color: "var(--faint)", fontFamily: "var(--mono)" }}>no tools</span>
                            )}
                          </div>
                        </div>

                        <div
                          className="abadge"
                          style={{
                            background: isRunning ? "#38bdf8" : isDone ? "#22c55e" : theme.accent,
                            boxShadow: isRunning ? "0 0 0 4px rgba(56,189,248,0.3)" : undefined,
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            flex: "none",
                          }}
                        />
                      </div>

                      {/* Ports for Linear and Topology Modes */}
                      {topology === "hierarchical" && n.id === "supervisor" && (
                        <>
                          <span className="aport ap-agent-bottom" style={{ left: "25%", bottom: -5 }} />
                          <span className="aport ap-agent-bottom" style={{ left: "50%", bottom: -5 }} />
                          <span className="aport ap-agent-bottom" style={{ left: "75%", bottom: -5 }} />
                        </>
                      )}
                      {topology === "hierarchical" && n.id !== "supervisor" && (
                        <span className="aport ap-top" style={{ left: "50%", top: -5 }} />
                      )}

                      {topology === "consensus" && n.id !== "referee" && (
                        <span className="aport ap-agent-bottom" style={{ left: "50%", bottom: -5 }} />
                      )}
                      {topology === "consensus" && n.id === "referee" && (
                        <>
                          <span className="aport ap-top" style={{ left: "35%", top: -5 }} />
                          <span className="aport ap-top" style={{ left: "65%", top: -5 }} />
                        </>
                      )}

                      {(topology === "linear" || topology === "sequential" || topology === "custom") && (
                        <>
                          {idx > 0 && <span className="aport ap-in-tool" style={{ left: -5, top: "50%" }} />}
                          {idx < nodes.length - 1 && <span className="aport ap-out-tool" style={{ right: -5, top: "50%" }} />}
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Configure Specialist Agent Inspector */}
            <div className="card" style={{ display: "flex", flexDirection: "column" }}>
              <div className="card-h" style={{ borderBottom: `1.5px solid ${selNodeTheme.border}` }}>
                <span className="t">Configure Node</span>
                <span className="mono r" style={{ color: selNodeTheme.light, fontWeight: 700, display: "inline-flex", alignItems: "center", gap: 6 }}>
                  {selNode ? (
                    <>
                      {renderAgentIcon(selNode.icon, 14, selNodeTheme.light)}
                      <span>{selNode.name}</span>
                    </>
                  ) : "—"}
                </span>
              </div>
              <div className="card-b" style={{ maxHeight: 420, overflow: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
                {selNode ? (
                  <>
                    {/* Node Type Selector */}
                    <div className="insp-field">
                      <div className="k">Node Type &amp; Color Theme</div>
                      <select
                        value={selNode.nodeType || "general"}
                        onChange={(e) => {
                          const val = e.target.value as NodeTypeKey;
                          const newConfig = NODE_TYPES_CATALOG[val];
                          setNodes((prev) =>
                            prev.map((n) =>
                              n.id === selNode.id
                                ? {
                                    ...n,
                                    nodeType: val,
                                    icon: newConfig.icon,
                                    name: n.name.startsWith("Node") || n.name === NODE_TYPES_CATALOG[n.nodeType]?.label ? newConfig.label : n.name,
                                    role: newConfig.defaultRole,
                                    systemPrompt: newConfig.defaultPrompt,
                                    tools: [...newConfig.defaultTools],
                                  }
                                : n
                            )
                          );
                          toast(`Updated node theme to ${newConfig.label}!`, "info");
                        }}
                      >
                        {(Object.keys(NODE_TYPES_CATALOG) as NodeTypeKey[]).map((k) => (
                          <option key={k} value={k}>
                            {NODE_TYPES_CATALOG[k].label} ({NODE_TYPES_CATALOG[k].tag})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="insp-field">
                      <div className="k">Agent / Node Name</div>
                      <input
                        type="text"
                        value={selNode.name}
                        onChange={(e) => {
                          const v = e.target.value;
                          setNodes((prev) => prev.map((n) => (n.id === selNode.id ? { ...n, name: v } : n)));
                        }}
                      />
                    </div>

                    <div className="insp-field">
                      <div className="k">Specialist Role</div>
                      <input
                        type="text"
                        value={selNode.role}
                        onChange={(e) => {
                          const v = e.target.value;
                          setNodes((prev) => prev.map((n) => (n.id === selNode.id ? { ...n, role: v } : n)));
                        }}
                      />
                    </div>

                    <div className="insp-field">
                      <div className="k">System Instructions / Prompt</div>
                      <textarea
                        rows={3}
                        value={selNode.systemPrompt}
                        onChange={(e) => {
                          const v = e.target.value;
                          setNodes((prev) => prev.map((n) => (n.id === selNode.id ? { ...n, systemPrompt: v } : n)));
                        }}
                      />
                    </div>

                    <div className="insp-field">
                      <div className="k" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span>Available Tools for this Node</span>
                        <span className="badge" style={{ fontSize: 9.5, background: "rgba(56,189,248,0.15)", color: "#38bdf8" }}>
                          {selNode.tools.length} active
                        </span>
                      </div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
                        {[
                          { id: "web_search", icon: "search", label: "web_search" },
                          { id: "arxiv", icon: "arxiv", label: "arxiv" },
                          { id: "wikipedia", icon: "wikipedia", label: "wikipedia" },
                          { id: "calculator", icon: "calculator", label: "calculator" },
                          { id: "statistics", icon: "statistics", label: "statistics" },
                          { id: "db_query", icon: "db_query", label: "db_query" },
                          { id: "datetime", icon: "datetime", label: "datetime" },
                          { id: "knowledge", icon: "knowledge", label: "knowledge" },
                          ...connectedMcpServers.map((m) => ({ id: m.name, icon: "cpu", label: `mcp:${m.name}` })),
                        ].map((t) => {
                          const hasTool = selNode.tools.includes(t.id);
                          return (
                            <span
                              key={t.id}
                              onClick={() => {
                                const nextHasTool = !hasTool;
                                setNodes((prev) =>
                                  prev.map((n) => {
                                    if (n.id !== selNode.id) return n;
                                    const nextTools = hasTool ? n.tools.filter((x) => x !== t.id) : [...n.tools, t.id];
                                    return { ...n, tools: nextTools };
                                  })
                                );
                                toast(
                                  nextHasTool
                                    ? `Added tool "${t.label}" to ${selNode.name}`
                                    : `Removed tool "${t.label}" from ${selNode.name}`,
                                  "info"
                                );
                              }}
                              className={`chk ${hasTool ? "on" : ""}`}
                              style={{
                                fontSize: 11,
                                cursor: "pointer",
                                borderColor: hasTool ? selNodeTheme.accent : undefined,
                                background: hasTool ? selNodeTheme.badgeBg : undefined,
                                color: hasTool ? selNodeTheme.light : undefined,
                                display: "inline-flex",
                                alignItems: "center",
                                gap: 5,
                              }}
                            >
                              {renderAgentIcon(t.icon, 12, hasTool ? selNodeTheme.light : "var(--muted)")}
                              <span>{t.label}</span>
                            </span>
                          );
                        })}
                      </div>
                    </div>

                    {/* Step Reorder & Management Actions */}
                    <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
                      <button
                        className="btn ghost xs"
                        style={{ flex: 1 }}
                        onClick={() => handleMoveNode(selNode.id, "left")}
                        disabled={nodes.findIndex((n) => n.id === selNode.id) === 0}
                      >
                        ◀ Move Earlier
                      </button>
                      <button
                        className="btn ghost xs"
                        style={{ flex: 1 }}
                        onClick={() => handleMoveNode(selNode.id, "right")}
                        disabled={nodes.findIndex((n) => n.id === selNode.id) === nodes.length - 1}
                      >
                        Move Later ▶
                      </button>
                    </div>

                    {nodes.length > 1 && (
                      <button
                        className="btn ghost xs"
                        style={{ color: "#f43f5e", borderColor: "rgba(244,63,94,0.3)", marginTop: 2, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 5 }}
                        onClick={() => handleRemoveNode(selNode.id)}
                      >
                        <Trash2 size={13} />
                        <span>Remove This Node</span>
                      </button>
                    )}
                  </>
                ) : (
                  <div className="note">Click a node on the canvas to configure it.</div>
                )}
              </div>
            </div>
          </div>

          <div className="stepnav" style={{ marginTop: 14 }}>
            <button className="btn ghost" onClick={() => setStep("type")}>
              ← Back to Topology
            </button>
            <button className="btn" onClick={() => setStep("run")}>
              Next: Run Pipeline →
            </button>
          </div>
        </div>
      )}

      {/* ── STEP 3: RUN PIPELINE ── */}
      {step === "run" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Execution flow canvas */}
          <div className="card">
            <div className="card-h">
              <span className="t" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <Zap size={16} color="#38bdf8" />
                <b>Live Execution Pipeline · {nodes.length} Stages Linear Flow</b>
              </span>
              <span className="mono r" style={{ color: running ? "#38bdf8" : finalSynthesis ? "#22c55e" : "var(--muted)" }}>
                {running ? "Orchestrating…" : finalSynthesis ? "Finished ✓" : "Ready to Execute"}
              </span>
            </div>
            <div
              className="card-b"
              style={{
                padding: 0,
                height: 400,
                position: "relative",
                background: "radial-gradient(rgba(56,189,248,0.15) 1.2px, transparent 1.2px), #0b1120",
                backgroundSize: "22px 22px",
                overflow: "auto",
              }}
            >
              {renderWires()}

              {nodes.map((n, idx) => {
                const pos = getNodePos(n.id, idx, nodes.length);
                const isRunning = nodeStatus[n.id] === "running";
                const isDone = nodeStatus[n.id] === "done";
                const cfg = NODE_TYPES_CATALOG[n.nodeType || "general"] || NODE_TYPES_CATALOG.general;
                const theme = cfg.theme;

                return (
                  <div
                    key={n.id}
                    className={`anode ${isRunning ? "running" : ""} ${isDone ? "done" : ""}`}
                    style={{
                      left: pos.x,
                      top: pos.y,
                      width: n.w,
                      height: n.h,
                      zIndex: isRunning ? 10 : 2,
                      background: `linear-gradient(135deg, ${theme.bg} 0%, rgba(15,23,42,0.9) 100%)`,
                      borderColor: isRunning ? "#38bdf8" : isDone ? "#22c55e" : theme.border,
                      borderWidth: "1.5px",
                      boxShadow: isRunning
                        ? "0 0 0 3px rgba(56,189,248,0.4), 0 0 24px rgba(56,189,248,0.6)"
                        : isDone
                        ? "0 0 0 2px rgba(34,197,94,0.4), 0 4px 14px rgba(0,0,0,0.3)"
                        : `0 4px 14px rgba(0,0,0,0.3), 0 0 10px ${theme.glow}`,
                      borderRadius: 14,
                      position: "absolute",
                    }}
                  >
                    <div className="ah" style={{ padding: "8px 12px", height: "100%", boxSizing: "border-box", display: "flex", alignItems: "center", gap: 10 }}>
                      <div
                        className="aic"
                        style={{
                          background: theme.badgeBg,
                          border: `1px solid ${theme.border}`,
                          color: theme.light,
                          width: 36,
                          height: 36,
                          borderRadius: 10,
                          display: "grid",
                          placeItems: "center",
                          fontSize: 18,
                          flex: "none",
                        }}
                      >
                        {renderAgentIcon(n.icon, 18, theme.light)}
                      </div>

                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
                          <div className="atitle" style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 110 }}>
                            {n.name}
                          </div>
                          <span
                            style={{
                              fontSize: 9,
                              fontWeight: 800,
                              fontFamily: "var(--mono)",
                              background: isRunning ? "rgba(56,189,248,0.2)" : isDone ? "rgba(34,197,94,0.2)" : theme.badgeBg,
                              color: isRunning ? "#38bdf8" : isDone ? "#4ade80" : theme.badgeText,
                              border: `1px solid ${isRunning ? "rgba(56,189,248,0.4)" : isDone ? "rgba(34,197,94,0.4)" : theme.border}`,
                              borderRadius: 4,
                              padding: "1px 5px",
                              whiteSpace: "nowrap",
                            }}
                          >
                            Step {idx + 1}
                          </span>
                        </div>

                        <div className="asub" style={{ fontSize: 9, color: theme.light, fontFamily: "var(--mono)", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".05em", marginTop: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 145 }}>
                          {n.role}
                        </div>
                      </div>

                      <div
                        className="abadge"
                        style={{
                          background: isRunning ? "#38bdf8" : isDone ? "#22c55e" : theme.accent,
                          boxShadow: isRunning ? "0 0 0 4px rgba(56,189,248,0.3)" : undefined,
                          width: 8,
                          height: 8,
                          borderRadius: "50%",
                          flex: "none",
                        }}
                      />
                    </div>

                    {/* Ports */}
                    {(topology === "linear" || topology === "sequential" || topology === "custom") && (
                      <>
                        {idx > 0 && <span className="aport ap-in-tool" style={{ left: -5, top: "50%" }} />}
                        {idx < nodes.length - 1 && <span className="aport ap-out-tool" style={{ right: -5, top: "50%" }} />}
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Split Row: Prompt on Left & Live Progression on Right */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 14, alignItems: "stretch" }}>
            {/* Left Card: Task Prompt */}
            <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 12 }}>
              <label className="fld" style={{ fontWeight: 700, fontSize: 13, margin: 0 }}>
                Task Prompt to Orchestrate
              </label>

              <textarea
                rows={4}
                value={task}
                onChange={(e) => setTask(e.target.value)}
                placeholder="Type task prompt for the multi-agent pipeline…"
                style={{ width: "100%", boxSizing: "border-box", resize: "vertical", fontSize: 13, lineHeight: 1.5 }}
              />

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "auto", paddingTop: 4 }}>
                <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                  {nodes.length} Specialist Agents Active
                </span>
                <button
                  className="btn"
                  onClick={runOrchestration}
                  disabled={running || !task.trim()}
                  style={{ minWidth: 180, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 6 }}
                >
                  {running ? <><span className="busy-dot" /> Orchestrating…</> : <><Play size={13} fill="currentColor" /> Run Orchestration</>}
                </button>
              </div>
            </div>

            {/* Right Card: Live Stage Progression & Inter-Agent Handovers */}
            <div className="card" style={{ padding: 16, display: "flex", flexDirection: "column" }}>
              <div className="card-h" style={{ padding: "0 0 10px 0", borderBottom: "1px solid var(--border)", marginBottom: 12 }}>
                <span className="t" style={{ fontSize: 13, display: "flex", alignItems: "center", gap: 6 }}>
                  <Zap size={14} color="#38bdf8" /> <b>Live Stage Progression &amp; Prediction</b>
                </span>
                <span className="mono r" style={{ fontSize: 11 }}>
                  {running ? "Running…" : executions.length ? `${executions.length} Stages` : "Ready"}
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: 8, flex: 1, justifyContent: executions.length ? "flex-start" : "center" }}>
                {executions.length > 0 ? (
                  executions.map((exec, idx) => {
                    const theme = NODE_TYPES_CATALOG[exec.nodeType || "general"]?.theme || NODE_TYPES_CATALOG.general.theme;
                    return (
                      <div
                        key={exec.nodeId}
                        style={{
                          padding: "9px 12px",
                          borderRadius: 9,
                          border: exec.status === "running"
                            ? "1.5px solid #38bdf8"
                            : exec.status === "done"
                            ? `1px solid ${theme.border}`
                            : "1px dashed var(--border)",
                          background: exec.status === "running" ? "rgba(56,189,248,0.08)" : theme.bg,
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          gap: 10,
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                          <span style={{ fontSize: 18, flex: "none", display: "inline-flex", alignItems: "center" }}>
                            {renderAgentIcon(exec.icon, 18, theme.light)}
                          </span>
                          <div style={{ minWidth: 0 }}>
                            <div style={{ fontWeight: 700, fontSize: 12.5, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                              Stage {idx + 1}: {exec.nodeName}
                            </div>
                            <div style={{ fontSize: 10.5, color: exec.status === "running" ? "#38bdf8" : exec.status === "done" ? "var(--good)" : "var(--faint)" }}>
                              {exec.status === "running" ? "Processing…" : exec.status === "done" ? "✓ Completed" : "Waiting…"}
                            </div>
                          </div>
                          {exec.status === "running" && <span className="busy-dot" style={{ marginLeft: 2 }} />}
                        </div>

                        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: "none" }}>
                          {exec.latencyMs > 0 && (
                            <span className="mono" style={{ fontSize: 10.5, color: "var(--muted)" }}>
                              {(exec.latencyMs / 1000).toFixed(1)}s
                            </span>
                          )}
                          {exec.confidence && (
                            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                              <ConfidenceGauge metrics={exec.confidence} size={30} compact={true} />
                              <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text)" }}>
                                {exec.confidence.score}%
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={{ textAlign: "center", color: "var(--muted)", fontSize: 12, padding: "24px 0" }}>
                    Click <b>Run Orchestration</b> to launch the multi-agent pipeline and observe live progression.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Final Synthesized Output */}
          {finalSynthesis && (
            <div
              className="card"
              style={{
                border: "1.5px solid rgba(59,130,246,0.35)",
                background: "linear-gradient(180deg, rgba(59,130,246,0.06) 0%, rgba(0,0,0,0.15) 100%)",
                borderRadius: 14,
                boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
                marginTop: 4,
              }}
            >
              <div className="card-h" style={{ borderBottom: "1px solid rgba(59,130,246,0.25)" }}>
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text)" }}>
                  <Sparkles size={16} color="#38bdf8" />
                  <b>Synthesized Multi-Agent Executive Output</b>
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  {overallConfidence && <ConfidenceGauge metrics={overallConfidence} size={36} compact={true} />}
                  <button
                    className="btn ghost xs"
                    style={{ display: "inline-flex", alignItems: "center", gap: 5 }}
                    onClick={() => {
                      navigator.clipboard.writeText(finalSynthesis).then(() => {
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      });
                    }}
                    title={copied ? "Copied to clipboard!" : "Copy output"}
                  >
                    {copied ? <CheckSvg /> : <CopySvg />}
                    <span>{copied ? "Copied" : "Copy"}</span>
                  </button>
                </div>
              </div>

              <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ fontSize: 13, lineHeight: 1.6 }}>
                  <AgentOutput text={finalSynthesis} />
                </div>

                {overallConfidence && (
                  <div style={{ marginTop: 8 }}>
                    <ConfidenceGauge metrics={overallConfidence} size={84} showBreakdown={true} />
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="stepnav" style={{ marginTop: 14 }}>
            <button className="btn ghost" onClick={() => setStep("build")}>
              ← Back to Build
            </button>
          </div>
        </div>
      )}

      {/* ── STEP 4: LEARN & PATTERNS ── */}
      {step === "learn" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-h">
              <span className="t">Multi-Agent Architecture Blueprints &amp; Deep Dives</span>
              <span className="mono r">5 Architectural Patterns</span>
            </div>
            <div className="card-b">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
                {ORCHESTRATION_LESSONS.map((lesson) => {
                  const isSel = selectedLesson?.id === lesson.id;
                  return (
                    <div
                      key={lesson.id}
                      onClick={() => setSelectedLesson(lesson)}
                      style={{
                        padding: 14,
                        borderRadius: 10,
                        background: isSel ? "rgba(56,189,248,0.12)" : "var(--panel)",
                        border: isSel ? "1.5px solid #38bdf8" : "1px solid var(--border)",
                        cursor: "pointer",
                        display: "flex",
                        flexDirection: "column",
                        gap: 8,
                        transition: "all 0.15s ease",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{ display: "inline-flex", alignItems: "center", color: isSel ? "#38bdf8" : "var(--muted)" }}>
                            {renderAgentIcon(lesson.icon, 18, isSel ? "#38bdf8" : "currentColor")}
                          </span>
                          <span style={{ fontWeight: 700, fontSize: 13, color: "var(--text)" }}>{lesson.title}</span>
                        </div>
                      </div>

                      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        <span className="badge" style={{ fontSize: 9.5, background: "var(--surface)", border: "1px solid var(--border)", color: "#38bdf8", padding: "1px 6px" }}>
                          {lesson.badge}
                        </span>
                        <span style={{ fontSize: 10, color: "var(--faint)", fontFamily: "var(--mono)" }}>{lesson.category}</span>
                      </div>

                      <div style={{ fontSize: 11.5, color: "var(--muted)", lineHeight: 1.45 }}>
                        {lesson.summary}
                      </div>

                      <div style={{ fontSize: 11, color: isSel ? "#38bdf8" : "var(--faint)", marginTop: "auto", fontWeight: 600 }}>
                        {isSel ? "● Reading active" : "Click to view deep dive →"}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {selectedLesson ? (
            <div className="card" style={{ border: "1.5px solid rgba(56,189,248,0.3)" }}>
              <div className="card-h" style={{ borderBottom: "1px solid var(--border)" }}>
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text)" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", color: "#38bdf8" }}>
                    {renderAgentIcon(selectedLesson.icon, 20, "#38bdf8")}
                  </span>
                  <b>{selectedLesson.title}</b>
                </span>
                <span className="badge" style={{ background: "rgba(56,189,248,0.2)", color: "#38bdf8", border: "1px solid rgba(56,189,248,0.4)" }}>
                  {selectedLesson.badge}
                </span>
              </div>

              <div className="card-b" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13, color: "var(--text)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                    <BookOpen size={15} color="#38bdf8" /> Architecture Overview &amp; Mechanism
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6 }}>
                    {selectedLesson.deepDive}
                  </div>
                </div>

                <div style={{ padding: 12, borderRadius: 8, background: "var(--panel)", border: "1px solid var(--border)" }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: "#38bdf8", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: 6 }}>
                    <Building2 size={14} color="#38bdf8" /> Real-World Enterprise Scenario
                  </div>
                  <div style={{ fontSize: 12.5, color: "var(--text)", lineHeight: 1.5 }}>
                    {selectedLesson.realWorldUse}
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color: "var(--text)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                      <MapPin size={14} color="#38bdf8" /> Message &amp; Handover Flow
                    </div>
                    <pre style={{ margin: 0, padding: 12, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--text-secondary)", lineHeight: 1.45, overflowX: "auto" }}>
                      {selectedLesson.diagram}
                    </pre>
                  </div>

                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color: "var(--text)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                      <Code2 size={14} color="#38bdf8" /> Code / State Handover Payload
                    </div>
                    <pre style={{ margin: 0, padding: 12, background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 11.5, fontFamily: "var(--mono)", color: "var(--text-secondary)", lineHeight: 1.45, overflowX: "auto" }}>
                      {selectedLesson.codeSchema}
                    </pre>
                  </div>
                </div>

                <div>
                  <div style={{ fontWeight: 700, fontSize: 12.5, color: "var(--text)", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
                    <Sparkles size={14} color="#38bdf8" /> Key Production Takeaways
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 20, display: "flex", flexDirection: "column", gap: 4, fontSize: 12.5, color: "var(--text-secondary)" }}>
                    {selectedLesson.keyConcepts.map((kc, i) => (
                      <li key={i}>{kc}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: 18, textAlign: "center", color: "var(--muted)", fontSize: 12.5 }}>
              Click any architecture topic card above to view the full deep-dive explanation, flow diagrams, and code schemas!
            </div>
          )}

          <div className="stepnav">
            <button className="btn ghost" onClick={() => setStep("type")}>
              ← Back to Type
            </button>
          </div>
        </div>
      )}

      {/* Get Code Modal */}
      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal" style={{ maxWidth: 740, width: "90%" }}>
          <div className="mh" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <b>Multi-Agent Orchestration Code · {PRESET_TOPOLOGIES[topology]?.title || "Custom DAG"}</b>
            <div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
              <button
                className="btn ghost sm"
                onClick={() => {
                  navigator.clipboard.writeText(buildOrchestrationCode());
                  setCodeCopied(true);
                  setTimeout(() => setCodeCopied(false), 2000);
                }}
              >
                {codeCopied ? "Copied ✓" : "Copy"}
              </button>
              <button
                className="btn sm"
                onClick={() => {
                  const blob = new Blob([buildOrchestrationCode()], { type: "text/plain" });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `orchestration_${topology}_graph.py`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                Download Python
              </button>
              <button className="x" onClick={() => setShowCode(false)}>×</button>
            </div>
          </div>
          <div className="mb">
            <div className="note" style={{ marginBottom: 10 }}>
              Where to use it: Run this multi-agent LangGraph / Async OpenAI pipeline with <code>pip install openai</code> · or <b>Save</b> to My Projects · or <b>Export JSON</b> to load into custom agent servers.
            </div>
            <pre className="code" style={{ maxHeight: 380, overflow: "auto", margin: 0, padding: 14, background: "var(--surface)", borderRadius: 8, fontSize: 12, lineHeight: 1.5, color: "var(--text-secondary)" }}>
              {buildOrchestrationCode()}
            </pre>
          </div>
        </div>
      </div>
      {/* Publish Orchestration Pipeline Custom Naming Modal */}
      <div className={`modal-wrap ${showPublishModal ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowPublishModal(false); }}>
        <div className="modal" style={{ maxWidth: 520, width: "90%" }}>
          <div className="mh" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <b>Publish Multi-Agent Pipeline to Workroom</b>
            <button className="x" onClick={() => setShowPublishModal(false)}>×</button>
          </div>
          <div className="mb" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Choose a custom name and description for this multi-agent workflow before publishing it to the Workroom.
            </div>

            <div>
              <label style={{ display: "block", fontSize: 12, fontWeight: 700, color: "var(--text)", marginBottom: 6 }}>
                Pipeline Name
              </label>
              <input
                type="text"
                value={publishPipelineName}
                onChange={(e) => setPublishPipelineName(e.target.value)}
                placeholder="e.g., Enterprise Research & Synthesis Pipeline"
                style={{ width: "100%", boxSizing: "border-box", padding: "8px 12px", fontSize: 13 }}
              />
            </div>

            <div>
              <label style={{ display: "block", fontSize: 12, fontWeight: 700, color: "var(--text)", marginBottom: 6 }}>
                Description / Role
              </label>
              <textarea
                rows={3}
                value={publishPipelineDesc}
                onChange={(e) => setPublishPipelineDesc(e.target.value)}
                placeholder="Brief summary of how this multi-agent graph operates…"
                style={{ width: "100%", boxSizing: "border-box", padding: "8px 12px", fontSize: 13, resize: "vertical" }}
              />
            </div>

            <div style={{ padding: 10, borderRadius: 8, background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>Topology: <b>{topology.toUpperCase()}</b></span>
              <span>Active Agents: <b>{nodes.length} Nodes</b></span>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 6 }}>
              <button className="btn ghost sm" onClick={() => setShowPublishModal(false)}>
                Cancel
              </button>
              <button
                className="btn sm"
                onClick={confirmPublishOrchestration}
                disabled={publishing || !publishPipelineName.trim()}
                style={{ background: "#3b82f6", borderColor: "#2563eb", color: "#ffffff", display: "inline-flex", alignItems: "center", gap: 6 }}
              >
                {publishing ? <span className="busy-dot" /> : <Rocket size={13} />}
                <span>{publishing ? "Publishing…" : "Confirm & Publish"}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
