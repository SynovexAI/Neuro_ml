"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  AGENT_TOOLS, buildKnowledge, reactSystemPrompt, parseReAct, formatFinalAnswer,
  type AgentTool, type ToolCtx,
} from "@/lib/agentTools";
import type { RagIndex } from "@/lib/ragUtils";
import NatAgentPanel from "./NatAgentPanel";
import { toast } from "@/lib/toast";

type AgentType = "react" | "workflow";
type Step = "type" | "build" | "run" | "learn";
type Provider = { id: string; provider: string; label: string | null; defaultModel: string };

export type LearnTopic = {
  id: string;
  title: string;
  category: string;
  icon: string;
  tagline: string;
  meaning: string;
  architectureOverview: string;
  flowSteps: { label: string; detail: string }[];
  applications: { where: string; how: string; impact: string }[];
  realExamples: { name: string; desc: string }[];
  codeExample: string;
  pitfallsAndMistakes: { mistake: string; fix: string }[];
  whereToExplore: string[];
  deepDive: string[];
  proTips: string[];
  quiz: {
    question: string;
    options: string[];
    answerIdx: number;
    explanation: string;
  };
  suggestedTemplate?: string;
};

const LEARN_MAP: LearnTopic[] = [
  // BRANCH 1: FUNDAMENTALS & DEFINITION
  {
    id: "what_is_agent",
    title: "What is an AI Agent?",
    category: "Fundamentals",
    icon: "🤖",
    tagline: "An autonomous software entity that perceives its environment, reasons using an LLM brain, and executes actions via tools to achieve goals.",
    meaning: "An AI Agent is more than just a chatbot. It is a goal-oriented autonomous software system that uses a Large Language Model (LLM) as its central cognitive brain. It receives environmental inputs (messages or data), plans multi-step actions, invokes external tools (APIs, code, databases), and continuously loops until it fulfills the user's objective.",
    architectureOverview: `Traditional software follows hardcoded IF/ELSE logic written by programmers. Simple LLM chatbots generate text response strings passively based on user prompts.

An AI Agent combines cognitive reasoning with active environmental tools:
1. Perception Engine: Receives user inputs, system context, and live observations.
2. Cognitive Brain (LLM): Analyzes goal, maintains memory, plans next logical step.
3. Action Engine (Tools): Executes real code, SQL queries, web searches, or REST API calls.
4. Feedback & Perception Loop: Evaluates tool outputs and decides whether to continue or emit final answer.`,
    flowSteps: [
      { label: "1. Environmental Perception", detail: "Agent receives user goal, system prompt instructions, and environment state." },
      { label: "2. Cognitive Reasoning (LLM)", detail: "LLM evaluates goal against memory and decides what information or action is needed." },
      { label: "3. Action Selection & Tool Dispatch", detail: "Agent formats tool invocation payload (e.g. { tool: 'search_db', query: 'customer #90' })." },
      { label: "4. Environment Execution", detail: "Backend runtime executes tool payload against real database, API, or calculator." },
      { label: "5. Observation Feedback & Final Answer", detail: "Tool result is appended to context; agent generates final grounded response." },
    ],
    applications: [
      { where: "Autonomous Customer Operations", how: "Agents resolve refund requests by querying databases, checking policy docs, and processing payments.", impact: "Automates 80% of support tickets end-to-end." },
      { where: "AI Software Engineering (Devin/Cursor)", how: "Agents read codebase, write code, run terminal commands, fix compiler errors iteratively.", impact: "Accelerates software development velocity by 3x." },
      { where: "Automated Financial Analysis", how: "Agents pull live market data, calculate financial metrics, and compile investment reports.", impact: "Eliminates manual spreadsheet work." },
    ],
    realExamples: [
      { name: "Devin by Cognition AI", desc: "The world's first autonomous AI software engineer that plans, writes, tests, and debugs code independently." },
      { name: "AutoGPT & CrewAI", desc: "Open-source agent frameworks that orchestrate multiple autonomous agents working together to achieve complex goals." },
      { name: "Intercom Fin Customer Bot", desc: "Enterprise support agent that resolves customer tickets autonomously using company knowledge bases." },
    ],
    codeExample: `// Core Definition of an AI Agent Loop (Python)

class AIAgent:
    def __init__(self, brain_model, tools, system_prompt):
        self.brain = brain_model
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory = []

    def run(self, user_goal):
        self.memory.append({"role": "user", "content": user_goal})
        
        while True:
            # 1. Think & Reason using LLM Brain
            response = self.brain.generate(self.system_prompt, self.memory)
            
            # 2. Check if final answer reached
            if response.has_final_answer:
                return response.final_answer
                
            # 3. Execute chosen Tool Action
            tool_output = self.tools.execute(response.action_name, response.action_args)
            
            # 4. Feedback loop: append Observation to memory
            self.memory.append({"role": "observation", "content": tool_output})`,
    pitfallsAndMistakes: [
      { mistake: "Confusing a Chatbot with an AI Agent", fix: "Chatbots only generate text. AI Agents use tools, execute code, and take real-world actions." },
      { mistake: "Omitting Environment Feedback Loops", fix: "Ensure tool outputs (observations) are always fed back into the LLM context." },
    ],
    whereToExplore: [
      "OpenAI Agent Documentation — guide to building autonomous tool agents",
      "LangChain & LangGraph — frameworks for building stateful agent loops",
      "This Agent Lab — build and test a ReAct Tool Agent live in Step 2 & 3",
    ],
    deepDive: [
      "Perception-Action Loop: The classic cybernetic feedback cycle applied to neural network architectures.",
      "Symbolic vs Connectionist AI: AI Agents combine connectionist neural LLM reasoning with symbolic software execution.",
    ],
    proTips: [
      "Always give your AI agent a clear persona and explicit tool boundaries in the system prompt.",
    ],
    quiz: {
      question: "What is the primary difference between a simple Chatbot and an AI Agent?",
      options: [
        "Chatbots run on GPUs while AI Agents run on CPUs",
        "Chatbots only generate text responses passively, while AI Agents autonomously reason, use tools, and take real-world actions to achieve goals",
        "Chatbots are written in Python while AI Agents are written in C++",
        "Chatbots do not require internet access"
      ],
      answerIdx: 1,
      explanation: "AI Agents are goal-driven autonomous systems equipped with tools, memory, and perception loops, whereas basic chatbots only produce passive text outputs."
    },
    suggestedTemplate: "support",
  },

  {
    id: "genai_vs_agent",
    title: "Generative AI vs. AI Agent",
    category: "Fundamentals",
    icon: "⚖️",
    tagline: "Generative AI creates content passively; AI Agents execute complex real-world goals autonomously.",
    meaning: "Generative AI refers to foundation models (LLMs like GPT-4, Claude, Llama) that predict text, code, or images based on training data. An AI Agent uses Generative AI as its 'brain' but surrounds it with tools, memory, planning algorithms, and execution runtimes to perform multi-step actions in the real world.",
    architectureOverview: `Comparison Matrix:

1. Input / Output:
   - Generative AI: Text Prompt → Generated Text Output.
   - AI Agent: Goal Directive → Reasoning → Tool Actions → State Changes → Final Answer.

2. Interactivity:
   - Generative AI: Passive (1-shot response generation).
   - AI Agent: Active (Multi-turn iterative loop with environment feedback).

3. Capabilities:
   - Generative AI: Writing essays, summarizing, answering general knowledge questions.
   - AI Agent: Buying tickets, updating SQL databases, writing and testing code, sending emails, issuing refunds.`,
    flowSteps: [
      { label: "1. Generative AI Phase (Text Generation)", detail: "User asks: 'Write a Python script to fetch stock prices.' Generative AI outputs code text." },
      { label: "2. AI Agent Phase (Active Execution)", detail: "Agent receives goal: 'Check Tesla stock price and email report to boss.'" },
      { label: "3. Tool Invocation Step 1", detail: "Agent calls stock_api tool to get live TSLA price ($210.50)." },
      { label: "4. Tool Invocation Step 2", detail: "Agent formats summary report and calls send_email tool." },
      { label: "5. Goal Completion", detail: "Agent verifies email sent and informs user: 'Tesla stock report ($210.50) emailed successfully.'" },
    ],
    applications: [
      { where: "Generative AI Use Case", how: "Drafting marketing copy, writing poems, summarizing long PDFs, translating languages.", impact: "Content creation acceleration." },
      { where: "AI Agent Use Case", how: "Automating customer support refunds, managing infrastructure, monitoring logs, conducting multi-step web research.", impact: "Full task automation." },
    ],
    realExamples: [
      { name: "ChatGPT (Generative AI Mode)", desc: "You ask ChatGPT for a recipe; it generates text listing ingredients and instructions." },
      { name: "ChatGPT Canvas / Operator (Agent Mode)", desc: "ChatGPT uses browser tools, runs Python code in a sandbox, and interacts with web pages to accomplish goals." },
    ],
    codeExample: `// Generative AI vs AI Agent Architectural Code Comparison

// 1. GENERATIVE AI (Passive 1-shot text generator)
const textOutput = await llm.generate("Write a SQL query to find top customers");
console.log(textOutput); // Prints query string. Does NOT execute query!

// 2. AI AGENT (Active autonomous executor)
const agent = new AIAgent({ llm, tools: [database_tool, email_tool] });
await agent.executeGoal("Find top 5 customers in DB and email them discount codes");
// Executes SQL query -> Retrieves emails -> Sends real emails!`,
    pitfallsAndMistakes: [
      { mistake: "Expecting Generative AI to execute actions without tools", fix: "Generative AI only produces text. You MUST give it function tools to take real actions." },
    ],
    whereToExplore: [
      "OpenAI API Docs — read about Chat Completions vs Function Calling & Assistants API",
      "This Agent Lab — compare raw text generation with ReAct Tool Agent execution",
    ],
    deepDive: [
      "Agency Scale: Level 0 (Text generation), Level 1 (Tool recommendation), Level 2 (Autonomous tool execution with human review), Level 3 (Fully autonomous agent).",
    ],
    proTips: [
      "Use Generative AI for creative synthesis; use AI Agents for task automation requiring real data.",
    ],
    quiz: {
      question: "Which of the following describes an AI Agent rather than plain Generative AI?",
      options: [
        "Generating a poem about space exploration",
        "Summarizing an article into 3 bullet points",
        "Executing a SQL query, analyzing sales figures, and automatically sending an email report to a manager",
        "Translating a sentence from English to French"
      ],
      answerIdx: 2,
      explanation: "Executing database queries, analyzing results, and sending real emails requires an AI Agent equipped with tools, memory, and an execution loop."
    },
    suggestedTemplate: "support",
  },

  // BRANCH 2: TYPES OF AI AGENTS
  {
    id: "types_of_agents",
    title: "Types of AI Agents",
    category: "Agent Types",
    icon: "🧩",
    tagline: "From simple reflex bots to autonomous ReAct agents, multi-step workflows, and collaborative multi-agent swarms.",
    meaning: "AI Agents are categorized based on their cognitive complexity and control architecture. Understanding these types allows developers to choose the right agent design pattern for their specific project requirements.",
    architectureOverview: `Key Agent Classifications:

1. Simple Reflex Agents: Act purely on current input rules (IF input matches X THEN action Y). No memory or historical context.
2. Model-Based Reflex Agents: Maintain internal state memory to track history and environment changes over time.
3. ReAct Autonomous Tool Agents: Use an LLM reasoning loop (Thought → Action → Observation) to dynamically decide paths at runtime.
4. Multi-Step Workflows (DAGs): Follow a fixed, pre-defined pipeline (Plan → Draft → Critique → Finalize) for predictable tasks.
5. Multi-Agent Swarms: Teams of specialized agents (e.g. Researcher + Writer + Editor) collaborating to solve massive goals.`,
    flowSteps: [
      { label: "1. Simple Reflex Agent", detail: "Input: 'Refund status' → Rule: Trigger lookup_refund tool immediately." },
      { label: "2. ReAct Autonomous Agent", detail: "Goal: 'Analyze sales drop' → Agent dynamically picks DB tool, then chart tool, then email tool." },
      { label: "3. Multi-Step Workflow", detail: "Step 1 (Plan) → Step 2 (Draft) → Step 3 (Critique) → Step 4 (Finalize)." },
      { label: "4. Multi-Agent Collaboration", detail: "Manager Agent delegates tasks to Specialist Agent A (Coder) and Specialist Agent B (Tester)." },
    ],
    applications: [
      { where: "ReAct Agents", how: "Complex troubleshooting, open-ended research, dynamic customer support.", impact: "Handles unpredictable user requests." },
      { where: "Workflow Agents", how: "Document publishing, code review pipelines, regulatory compliance reports.", impact: "Guarantees 100% predictable execution." },
      { where: "Multi-Agent Swarms", how: "Full-stack software development (PM agent, Frontend agent, Backend agent, QA agent).", impact: "Solves enterprise-scale problems." },
    ],
    realExamples: [
      { name: "CrewAI Multi-Agent Framework", desc: "Allows developers to create teams of agents (e.g. Researcher Crew + Writer Crew) with distinct roles." },
      { name: "LangChain ReAct Agent", desc: "Standard single-agent ReAct loop used for tool calling and web search." },
    ],
    codeExample: `// Agent Type Architecture Comparison (TypeScript)

// 1. ReAct Tool Agent (Dynamic Reasoning Loop)
const reactAgent = new ReActAgent({
  model: "gpt-4o",
  tools: [calculator, search_db]
});

// 2. Multi-Step Workflow Agent (Fixed Pipeline)
const workflowAgent = new WorkflowAgent({
  steps: [
    { name: "Outline", instruction: "Create 3-point outline" },
    { name: "Draft", instruction: "Write article from outline" },
    { name: "Edit", instruction: "Proofread and polish" }
  ]
});`,
    pitfallsAndMistakes: [
      { mistake: "Over-engineering simple tasks with Multi-Agent Swarms", fix: "Start with a single ReAct or Workflow agent before introducing multi-agent complexity." },
    ],
    whereToExplore: [
      "CrewAI Documentation — learn multi-agent team orchestration",
      "This Agent Lab — toggle between ReAct Tool Agent and Multi-step Workflow in Step 1",
    ],
    deepDive: [
      "Agent Orchestration Patterns: Supervisor pattern (one manager agent routes tasks) vs Hierarchical swarms.",
    ],
    proTips: [
      "Use ReAct agents for open-ended problem solving; use Workflows for structured business processes.",
    ],
    quiz: {
      question: "Which agent architecture should you choose for a structured 4-step content publishing pipeline (Outline -> Draft -> Critique -> Finalize)?",
      options: [
        "Simple Reflex Agent",
        "Multi-Step Deterministic Workflow Agent",
        "Random Sampling Agent",
        "Vector Embedding Agent"
      ],
      answerIdx: 1,
      explanation: "Deterministic Workflows are designed specifically for pre-defined, sequential step pipelines where execution order is fixed and predictable."
    },
    suggestedTemplate: "support",
  },

  // BRANCH 3: EXECUTION FLOW (MESSAGE TO ACTION)
  {
    id: "message_understanding",
    title: "How Agents Understand Messages",
    category: "Execution Flow",
    icon: "🧠",
    tagline: "Tokenization, system prompt priming, intent parsing, and LLM context window assembly.",
    meaning: "When a user sends a message to an AI Agent, the raw text is not read as human words. The input is tokenized into numerical IDs, combined with System Prompt rules and conversation history, and passed through Transformer self-attention layers to parse user intent and extract tool arguments.",
    architectureOverview: `Message Processing Steps:

1. Tokenization: Text is converted into token IDs using Byte-Pair Encoding (BPE) (e.g. "AI Agent" → [9552, 18241]).
2. Context Window Assembly: System Prompt (Rules) + Memory (Past Chat) + Current User Message are concatenated into a single context array.
3. Self-Attention Pass: Transformer layers compute attention weights between tokens to parse intent (e.g. identifying that "book a flight" requires travel_api).
4. Intent & Entity Extraction: LLM determines whether the user needs a direct text answer OR a tool call with specific arguments.`,
    flowSteps: [
      { label: "1. Raw Message Arrival", detail: "User types: 'Calculate 15% tip on $85 bill.'" },
      { label: "2. BPE Tokenization", detail: "Text is tokenized into numerical integers representing sub-word units." },
      { label: "3. System Priming", detail: "System prompt instructions ('You are a calculator agent...') are prepended." },
      { label: "4. Intent Recognition", detail: "LLM self-attention recognizes math intent and identifies parameters: amount=85, tip=0.15." },
      { label: "5. Action Payload Generation", detail: "LLM emits tool request: Action: calculator | Input: 85 * 0.15." },
    ],
    applications: [
      { where: "Intent Routing", how: "Classify incoming user messages to route them to support, billing, or sales sub-agents.", impact: "Increases routing accuracy to 98%." },
      { where: "Entity Extraction", how: "Extract customer names, order IDs, and dates automatically from unstructured user emails.", impact: "Eliminates manual data entry." },
    ],
    realExamples: [
      { name: "OpenAI Tokenizer (tiktoken)", desc: "OpenAI's fast BPE tokenizer used to split text into tokens before passing to GPT models." },
      { name: "Rasa / Dialogflow Intent Parsing", desc: "Traditional NLU intent parsers replaced by modern LLM zero-shot intent recognition." },
    ],
    codeExample: `// Tokenization & Context Window Assembly (Python)

import tiktoken

# 1. Tokenize Text
encoder = tiktoken.encoding_for_model("gpt-4o")
tokens = encoder.encode("How does an AI agent understand messages?")
print(f"Token Count: {len(tokens)}, Token IDs: {tokens[:5]}...")

# 2. Context Window Payload Array
messages = [
    {"role": "system", "content": "You are an AI Agent Tutor. Answer concisely."},
    {"role": "user", "content": "What is an embedding?"}
]`,
    pitfallsAndMistakes: [
      { mistake: "Exceeding Context Window Token Limit", fix: "Monitor total token counts (system + memory + query) and truncate old chat history." },
    ],
    whereToExplore: [
      "OpenAI Tokenizer Web Tool — paste text to visually inspect token boundaries",
      "This Agent Lab — view token metrics in real-time after running agent requests in Step 3",
    ],
    deepDive: [
      "Self-Attention Mechanism: Query (Q), Key (K), Value (V) matrices calculate token relationships across long contexts.",
    ],
    proTips: [
      "Keep system prompt formatting clean so the LLM easily distinguishes rules from user query inputs.",
    ],
    quiz: {
      question: "What happens to raw text when a user sends a message to an LLM-powered AI Agent?",
      options: [
        "It is directly converted into SQL queries",
        "It is split into numerical token IDs via tokenization and combined with System Prompt rules before self-attention processing",
        "It is saved into a spreadsheet file",
        "It is translated into binary C++ code"
      ],
      answerIdx: 1,
      explanation: "Text is tokenized into numerical token IDs and assembled alongside the system prompt and conversation history into the LLM context window."
    },
    suggestedTemplate: "support",
  },

  {
    id: "full_execution_flow",
    title: "Complete Agent Execution Flow",
    category: "Execution Flow",
    icon: "🔄",
    tagline: "The end-to-end trace: User Msg → Tokenization → LLM Reasoning → Tool Choice → Execution → Observation → Answer.",
    meaning: "The complete agent execution flow is the step-by-step lifecycle of a request inside an agent framework. It traces how a user prompt moves from initial text input through LLM cognitive reasoning, tool execution, observation injection, and final response synthesis.",
    architectureOverview: `The complete agent execution flow is a closed-loop state machine combining stochastic LLM reasoning with deterministic code execution.

Key Lifecycle Phases:
1. Ingestion & Tokenization: The user query is tokenized (BPE) and combined with system prompt instructions and conversation history into an assembled context window.
2. Cognitive Inference Pass: The LLM processes context via Transformer self-attention. It evaluates whether the query requires direct text synthesis OR an external tool execution.
3. Decision Branching:
   - Path A (Direct Answer): If no external data is required, the LLM generates a text output and completes the request turn.
   - Path B (Tool Action): If live facts or software capabilities are needed, the LLM emits a tool payload (e.g. { tool: "calculator", input: "50 * 1.08" }).
4. Application Runtime Dispatch: The host application intercepts the payload, executes the real function (Python math, SQL query, API call), and receives a raw result string.
5. Observation Feedback Injection: The result is injected back into the LLM context as an "Observation" turn. The LLM re-evaluates context until a final answer is produced.`,
    flowSteps: [
      { label: "1. User Message Ingestion", detail: "User submits query: 'Check stock price of Apple and calculate 10 shares value.'" },
      { label: "2. Prompt Assembly & Tokenization", detail: "Context window assembled with System Prompt, Tool Schemas, and User Query." },
      { label: "3. LLM Reasoning Pass 1", detail: "LLM emits Action: stock_lookup | Input: { ticker: 'AAPL' }." },
      { label: "4. Runtime Tool Execution 1", detail: "Backend calls Financial API, receives Observation: { price: 220.00 }." },
      { label: "5. LLM Reasoning Pass 2", detail: "LLM sees AAPL price $220. Emits Action: calculator | Input: 10 * 220." },
      { label: "6. Runtime Tool Execution 2", detail: "Calculator tool returns Observation: 2200." },
      { label: "7. Final Answer Synthesis", detail: "LLM synthesizes: 'Apple (AAPL) is $220.00/share. 10 shares equal $2,200.00.'" },
    ],
    applications: [
      { where: "Agent Debugging & Observability", how: "Trace execution flow step-by-step to catch loop failures, tool errors, or prompt bugs.", impact: "Reduces agent debugging time by 90%." },
    ],
    realExamples: [
      { name: "LangSmith by LangChain", desc: "Production tracing platform that visualizes every step of an agent's execution flow in real-time." },
      { name: "Arize Phoenix", desc: "Open-source AI observability platform for tracing LLM agent execution graphs and tool payloads." },
    ],
    codeExample: `// End-to-End Execution Flow Observer (TypeScript)

async function traceAgentExecution(userQuery: string) {
  console.log("[1. User Msg Ingested]:", userQuery);

  const context = buildContextWindow(systemPrompt, memory, userQuery);
  console.log("[2. Context Assembled]: Token count =", countTokens(context));

  const decision = await llmInference(context);

  if (decision.type === "TOOL_CALL") {
    console.log(\`[3. LLM Action Emitted]: \${decision.toolName} (\${JSON.stringify(decision.args)})\`);
    
    const result = await executeToolBackend(decision.toolName, decision.args);
    console.log("[4. Observation Returned]:", result);
    
    // Loop back with observation
    return await traceAgentExecution(\`Observation from \${decision.toolName}: \${result}\`);
  } else {
    console.log("[5. Final Answer Emitted]:", decision.text);
    return decision.text;
  }
}`,
    pitfallsAndMistakes: [
      { mistake: "Not logging intermediate Thought and Action steps", fix: "Always maintain execution trace logs so you can inspect why an agent made a specific tool choice." },
    ],
    whereToExplore: [
      "LangSmith Observability — visualize agent execution trees live",
      "This Agent Lab — inspect the Reasoning Trace panel in Step 3 while running an agent",
    ],
    deepDive: [
      "Agent State Graphs: Modeling agent execution as directed graphs (nodes = actions, edges = state transitions).",
    ],
    proTips: [
      "In production, set strict timeouts on tool execution steps so slow APIs do not hang the agent flow.",
    ],
    quiz: {
      question: "In a ReAct agent execution flow, what happens after a tool action is executed by the backend runtime?",
      options: [
        "The agent immediately crashes",
        "The tool output (Observation) is appended back into the LLM context window, and the LLM performs another reasoning pass",
        "The user's computer reboots",
        "The vector database is deleted"
      ],
      answerIdx: 1,
      explanation: "The result of tool execution is returned as an 'Observation', which is fed back into the context so the LLM can evaluate whether it needs another tool or can give the final answer."
    },
    suggestedTemplate: "analyst",
  },

  {
    id: "why_llm_and_api",
    title: "Why LLMs & APIs Are Used",
    category: "Execution Flow",
    icon: "⚡",
    tagline: "LLMs provide the cognitive reasoning brain; APIs provide the hands to interact with software systems.",
    meaning: "An AI Agent requires two complementary technologies: 1) Large Language Models (LLMs) like GPT-4o or Claude 3.5 to serve as the reasoning engine for understanding natural language and making decisions, and 2) Application Programming Interfaces (APIs) to serve as external 'hands' for fetching live data and taking real-world software actions.",
    architectureOverview: `The Synergistic Architecture:

1. The LLM (Cognitive Brain):
   - Handles natural language understanding, intent classification, entity extraction, planning, and synthesis.
   - Cannot natively access live internet data, run database queries, or send emails on its own.

2. APIs & Tools (Hands & Feet):
   - Expose software capabilities (REST APIs, GraphQL, SQL connectors, Python runtimes) as callable functions.
   - Execute deterministic logic and return factual data back to the LLM.

Why Both Are Essential:
Without an LLM, software requires hardcoded rules and cannot understand free-form human language.
Without APIs, an LLM is isolated in a text sandbox unable to interact with real-world databases or services.`,
    flowSteps: [
      { label: "1. Human Intent Expression", detail: "User speaks in natural language: 'Book a conference room for 3 PM tomorrow.'" },
      { label: "2. LLM Reasoning (Brain)", detail: "LLM parses intent: Action: book_room | Params: { time: '15:00', date: 'tomorrow' }." },
      { label: "3. REST API Call (Hands)", detail: "Application runtime fires HTTP POST request to Office 365 / Google Calendar REST API." },
      { label: "4. API Response", detail: "Calendar API returns HTTP 200 OK: { status: 'CONFIRMED', room: 'Room 4B' }." },
      { label: "5. LLM Response Synthesis", detail: "LLM translates raw API JSON into friendly message: 'Room 4B booked for 3 PM tomorrow!'" },
    ],
    applications: [
      { where: "Enterprise System Integration", how: "Connect LLMs to SAP, Salesforce, Jira, and ServiceNow APIs.", impact: "Unifies corporate software behind a natural language AI interface." },
      { where: "E-Commerce Automations", how: "Connect LLMs to Shopify API for order tracking, inventory updates, and refunds.", impact: "Automates customer service workflows." },
    ],
    realExamples: [
      { name: "OpenAI REST API", desc: "The cloud API service providing access to GPT-4o reasoning models for developers worldwide." },
      { name: "Stripe Payment API", desc: "Exposed as an AI-callable tool so agents can process credit card payments securely." },
    ],
    codeExample: `// Combining OpenAI LLM Brain with REST API Hands (Python)

import openai
import requests

# 1. LLM Brain decides tool call
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is current Bitcoin price in USD?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}}
        }
    }]
)

# 2. REST API Hands execute request
if response.choices[0].message.tool_calls:
    # Call real CoinGecko REST API
    api_res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
    btc_price = api_res.json()["bitcoin"]["usd"]
    print(f"Live BTC Price from REST API: \${btc_price}")`,
    pitfallsAndMistakes: [
      { mistake: "Hardcoding API logic inside system prompts instead of function calls", fix: "Expose APIs as structured function schemas with JSON parameters." },
    ],
    whereToExplore: [
      "OpenAI API Platform — explore model endpoints and API keys",
      "Postman API Network — browse public REST APIs to connect to your agents",
    ],
    deepDive: [
      "API Rate Limiting & Retry Strategies: Handling HTTP 429 Rate Limits and 503 Gateway errors in agent tool runtimes.",
    ],
    proTips: [
      "Always secure API keys in environment variables (.env) — never hardcode them in agent scripts.",
    ],
    quiz: {
      question: "Why are REST APIs combined with LLMs when building AI Agents?",
      options: [
        "Because LLMs cannot read English without APIs",
        "Because LLMs provide the cognitive reasoning brain, while REST APIs provide the real-world hands to fetch live data and perform actions in external software systems",
        "To format CSS colors",
        "To reduce computer monitor brightness"
      ],
      answerIdx: 1,
      explanation: "LLMs handle natural language reasoning and intent, while REST APIs provide the functional capability to interact with real-world databases, web services, and applications."
    },
    suggestedTemplate: "analyst",
  },

  // BRANCH 4: KNOWLEDGE, CHUNKS & EMBEDDINGS
  {
    id: "chunks_and_embeddings",
    title: "Document Chunks & Vector Embeddings",
    category: "Knowledge & RAG",
    icon: "📚",
    tagline: "Overcoming LLM context limits by splitting documents into chunks and searching high-dimensional vector spaces.",
    meaning: "Large documents (PDFs, manuals, contracts) cannot fit entirely into an LLM context window. To solve this, documents are broken down into smaller text 'chunks' (250–500 tokens). Each chunk is processed by an Embedding Model to generate a numerical vector (a list of 1536 floating-point numbers) representing its semantic meaning in vector space.",
    architectureOverview: `Vector RAG Mechanics:

1. Text Chunking:
   - Raw PDF text is sliced into manageable chunks (e.g. 400 tokens per chunk).
   - 10-20% token overlap is added between consecutive chunks so context is not cut across sentence boundaries.

2. Vector Embedding Generation:
   - Text Chunk: "Refunds are processed within 5 business days."
   - Embedding Model (text-embedding-3-small): Encodes text into a 1536-dimensional vector: [-0.012, 0.045, 0.089, ..., 0.003].
   - Concepts with similar meanings (e.g., "refund" and "money back") are positioned close together in vector space.

3. Semantic Similarity Search:
   - User Query: "How long does it take to get my money back?"
   - Query Vectorized → Vector DB measures Cosine Distance against all chunk vectors.
   - Top K (e.g. 3) closest chunks are retrieved and injected into the LLM context.`,
    flowSteps: [
      { label: "1. Document Parsing & Text Slicing", detail: "PDF document parsed into 400-token chunks with 50-token overlap." },
      { label: "2. Dense Embedding Vector Encoding", detail: "OpenAI text-embedding-3-small generates 1536-dimensional float vector array per chunk." },
      { label: "3. Vector Database Indexing", detail: "Vectors and metadata stored in vector database (ChromaDB, Pinecone, FAISS)." },
      { label: "4. Query Embedding & Cosine Search", detail: "User query embedded; Cosine Similarity calculates spatial distance between vectors." },
      { label: "5. Grounded Prompt Context Injection", detail: "Top 3 nearest document text chunks injected into LLM prompt context." },
    ],
    applications: [
      { where: "Corporate Knowledge Search", how: "Search 10,000+ internal HR and IT policy documents instantly.", impact: "Eliminates manual file searching." },
      { where: "Legal & Regulatory Compliance", how: "Query thousands of legal contracts to locate specific indemnification clauses.", impact: "Reduces contract review time from days to seconds." },
    ],
    realExamples: [
      { name: "Pinecone Vector Database", desc: "Managed cloud vector database built specifically for high-speed similarity search at scale." },
      { name: "OpenAI text-embedding-3-small", desc: "OpenAI's industry-standard text embedding model producing 1536-dimensional vectors." },
    ],
    codeExample: `// Document Chunking & Vector Embedding Pipeline (Python)

import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Chunk Document with Overlap
raw_text = "Acme Corp Policy Manual... [100 pages of text]"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)
print(f"Generated {len(chunks)} document chunks.")

# 2. Generate Vector Embedding for Chunk #1
embedding_res = openai.embeddings.create(
    model="text-embedding-3-small",
    input=chunks[0]
)
vector = embedding_res.data[0].embedding
print(f"Vector Dimensions: {len(vector)} (e.g. {vector[:3]}...)")`,
    pitfallsAndMistakes: [
      { mistake: "Chunk sizes too large (> 2000 tokens)", fix: "Keep chunk size between 250–500 tokens so similarity search pinpoints specific facts." },
      { mistake: "Omitting chunk overlap", fix: "Always use 10–20% token overlap so sentence context is not severed between chunks." },
    ],
    whereToExplore: [
      "Pinecone Vector Embedding Guide — interactive visualization of multi-dimensional vector space",
      "ChromaDB Open-Source DB — run a vector database locally in Python for free",
      "This Agent Lab — upload files in Step 2 to test RAG chunk retrieval live",
    ],
    deepDive: [
      "Cosine Similarity vs Euclidean Distance: Cosine measures angle between vectors regardless of length; Euclidean measures straight-line distance.",
    ],
    proTips: [
      "Always attach metadata (filename, page number) to chunk vectors so the LLM can cite source pages.",
    ],
    quiz: {
      question: "Why do we split long documents into 'chunks' before converting them into vector embeddings for RAG?",
      options: [
        "To make the files take up more disk space",
        "Because LLMs have context token limits, and smaller chunks allow vector search to pinpoint exact relevant paragraphs matching the query",
        "To encrypt the documents",
        "To change the font style of the text"
      ],
      answerIdx: 1,
      explanation: "Chunking splits large documents into focused paragraphs so vector search can retrieve only the specific paragraphs relevant to the user's prompt, fitting cleanly within context limits."
    },
    suggestedTemplate: "research",
  },

  // BRANCH 5: GENERAL IMPLEMENTATION STEPS
  {
    id: "implementation_steps",
    title: "5 General Steps to Build an AI Agent",
    category: "Implementation",
    icon: "🚀",
    tagline: "The step-by-step developer methodology: Persona → Tools → Cognitive Loop → RAG Knowledge → Guardrails.",
    meaning: "Building a production-ready AI Agent follows a structured 5-step engineering methodology. Skipping any of these foundational steps leads to unreliable behavior, hallucinations, security vulnerabilities, or uncontrolled execution loops.",
    architectureOverview: `The 5-Step Developer Blueprint:

Step 1: Define Persona & System Instructions
- Establish identity, capacity, negative constraints, and output formatting rules.

Step 2: Register Tools & Function Schemas
- Expose external code, REST APIs, calculators, or DB connectors as callable functions with JSON parameters.

Step 3: Choose Cognitive Loop Architecture
- Select ReAct loop (for open-ended autonomous tasks) or Deterministic Workflow (for fixed step pipelines).

Step 4: Attach Knowledge RAG Store (Optional)
- Ingest private PDFs, manuals, or DB records into a vector database for semantic context retrieval.

Step 5: Implement Guardrails & Human-in-the-Loop (HITL)
- Add safety checks, token limits, iteration caps, and human approval checkpoints for high-risk actions.`,
    flowSteps: [
      { label: "Step 1 — Persona & System Prompt", detail: "Write system instructions defining role, boundaries, rules, and output syntax." },
      { label: "Step 2 — Tool & API Registration", detail: "Define tool functions with clear names, active descriptions, and parameter schemas." },
      { label: "Step 3 — Cognitive Control Loop", detail: "Implement ReAct loop or Workflow DAG pipeline with state memory management." },
      { label: "Step 4 — Vector RAG Integration", detail: "Connect vector database index to supply private document context." },
      { label: "Step 5 — Security & HITL Guardrails", detail: "Set iteration caps (max 6), token limits, and human approval triggers for high-risk tools." },
    ],
    applications: [
      { where: "Enterprise Agent Engineering", how: "Follow this 5-step methodology to build production AI agents for finance, support, and IT operations.", impact: "Ensures production reliability and safety." },
    ],
    realExamples: [
      { name: "LangGraph Architecture Blueprint", desc: "LangChain's standard developer workflow for building production agent graphs following these 5 steps." },
    ],
    codeExample: `// Complete 5-Step AI Agent Developer Blueprint (TypeScript)

// STEP 1: Persona & System Prompt
const systemPrompt = "You are a Customer Support Agent. Use tools when facts are needed.";

// STEP 2: Tools & Function Schemas
const tools = [calculatorTool, databaseLookupTool];

// STEP 3: Cognitive Control Loop (ReAct)
const agentLoop = new ReActEngine({ model: "gpt-4o", systemPrompt, tools });

// STEP 4: Vector RAG Knowledge Integration
agentLoop.attachKnowledgeBase(vectorDbIndex);

// STEP 5: Security & Human Approval Guardrails
agentLoop.setGuardrails({
  maxIterations: 6,
  humanApprovalRequired: ["issue_refund", "delete_record"]
});

// Run Agent
await agentLoop.execute("Help customer #402 get a refund for damaged item.");`,
    pitfallsAndMistakes: [
      { mistake: "Deploying an autonomous agent without Step 5 (Guardrails & HITL)", fix: "Always configure max iteration caps and human approval for financial or data modification tools." },
    ],
    whereToExplore: [
      "This Agent Lab — use the Build tab to configure Step 1 (Persona), Step 2 (Tools), Step 3 (Run), and Step 4 (Learn)",
      "OpenAI Assistants API Quickstart — building agents via official SDKs",
    ],
    deepDive: [
      "Agent Testing & Evaluation (Evals): Using automated eval suites (Ragas, TruLens) to benchmark agent accuracy across 100+ test cases.",
    ],
    proTips: [
      "Build and test each step individually before combining them into a full autonomous agent loop.",
    ],
    quiz: {
      question: "What is the final essential step (Step 5) in building a production-grade AI Agent?",
      options: [
        "Buying a faster computer monitor",
        "Implementing Guardrails & Human-in-the-Loop (HITL) safety checks to enforce token limits and human review on high-risk actions",
        "Writing a blog post",
        "Deleting the source code"
      ],
      answerIdx: 1,
      explanation: "Step 5 (Guardrails & HITL) ensures that the agent operates safely, preventing infinite loops, budget overruns, or unapproved destructive actions."
    },
    suggestedTemplate: "support",
  },
];

// Modern Tech SVG Icon Components (Replacing Emojis)
const BotSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <rect x="3" y="11" width="18" height="10" rx="2" />
    <circle cx="12" cy="5" r="2" />
    <path d="M12 7v4" />
    <line x1="8" y1="16" x2="8.01" y2="16" strokeWidth="3" />
    <line x1="16" y1="16" x2="16.01" y2="16" strokeWidth="3" />
  </svg>
);

const PuzzleSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M19.439 7.85c-.049-.322.059-.648.289-.878l1.564-1.564a2.121 2.121 0 0 0-3-3l-1.564 1.564c-.23.23-.556.338-.878.289a3.001 3.001 0 0 0-3.41 3.41c.049.322-.059.648-.289.878l-1.564 1.564a2.121 2.121 0 0 0 3 3l1.564-1.564c.23-.23.556-.338.878-.289a3.001 3.001 0 0 0 3.41-3.41z" />
    <path d="M4 14.5A2.5 2.5 0 0 0 6.5 17H9v2.5a2.5 2.5 0 0 0 5 0V17h2.5a2.5 2.5 0 0 0 0-5H14V9.5a2.5 2.5 0 0 0-5 0V12H6.5A2.5 2.5 0 0 0 4 14.5z" />
  </svg>
);

const RefreshSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <polyline points="23 4 23 10 17 10" />
    <polyline points="1 20 1 14 7 14" />
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
  </svg>
);

const BookSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
  </svg>
);

const BuildSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <rect x="2" y="6" width="20" height="8" rx="1" />
    <path d="M17 14v7" />
    <path d="M7 14v7" />
    <path d="M17 3v3" />
    <path d="M7 3v3" />
  </svg>
);

const ZapSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

const BrainSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04" />
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0-.34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04" />
  </svg>
);

const TargetSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <circle cx="12" cy="12" r="10" />
    <circle cx="12" cy="12" r="6" />
    <circle cx="12" cy="12" r="2" />
  </svg>
);

const ChartSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
);

const GraduationSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
    <path d="M6 12v5c3 3 9 3 12 0v-5" />
  </svg>
);

const CrownSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M2 4l3 12h14l3-12-6 7-4-7-4 7-6-7z" />
  </svg>
);

const UsersSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const MapSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
    <line x1="8" y1="2" x2="8" y2="18" />
    <line x1="16" y1="6" x2="16" y2="22" />
  </svg>
);

const AlertSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const ScaleSvg = ({ size = 16, color = "currentColor" }: { size?: number; color?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle" }}>
    <path d="M12 3v18" />
    <path d="M3 7h18" />
    <path d="M6 7l-3 7h6l-3-7z" />
    <path d="M18 7l-3 7h6l-3-7z" />
  </svg>
);

const CURRICULUM_CATEGORIES = [
  {
    id: "Fundamentals",
    phase: "Phase 01",
    label: "Fundamentals & Definition",
    getSvgIcon: (color: string) => <BotSvg size={15} color={color} />,
    color: "#6366f1",
    bg: "rgba(99, 102, 241, 0.08)",
    border: "rgba(99, 102, 241, 0.35)",
    tagline: "Core agent loop & LLM brain comparison",
    getTopics: () => LEARN_MAP.filter((t) => t.category === "Fundamentals")
  },
  {
    id: "Agent Types",
    phase: "Phase 02",
    label: "Types of AI Agents",
    getSvgIcon: (color: string) => <PuzzleSvg size={15} color={color} />,
    color: "#a855f7",
    bg: "rgba(168, 85, 247, 0.08)",
    border: "rgba(168, 85, 247, 0.35)",
    tagline: "Simple reflex, ReAct & multi-agent swarms",
    getTopics: () => LEARN_MAP.filter((t) => t.category === "Agent Types")
  },
  {
    id: "Execution Flow",
    phase: "Phase 03",
    label: "Execution & Message Flow",
    getSvgIcon: (color: string) => <RefreshSvg size={15} color={color} />,
    color: "#38bdf8",
    bg: "rgba(56, 189, 248, 0.08)",
    border: "rgba(56, 189, 248, 0.35)",
    tagline: "Token window assembly & tool dispatch",
    getTopics: () => LEARN_MAP.filter((t) => t.category === "Execution Flow")
  },
  {
    id: "Knowledge & RAG",
    phase: "Phase 04",
    label: "Chunks, Embeddings & RAG",
    getSvgIcon: (color: string) => <BookSvg size={15} color={color} />,
    color: "#10b981",
    bg: "rgba(16, 185, 129, 0.08)",
    border: "rgba(16, 185, 129, 0.35)",
    tagline: "Text chunking, vector DBs & similarity search",
    getTopics: () => LEARN_MAP.filter((t) => t.category === "Knowledge & RAG" || t.category === "Knowledge")
  },
  {
    id: "Implementation",
    phase: "Phase 05",
    label: "5-Step Implementation",
    getSvgIcon: (color: string) => <BuildSvg size={15} color={color} />,
    color: "#f59e0b",
    bg: "rgba(245, 158, 11, 0.08)",
    border: "rgba(245, 158, 11, 0.35)",
    tagline: "System prompts, tool schemas & guardrails",
    getTopics: () => LEARN_MAP.filter((t) => t.category === "Implementation")
  },
];

const TYPES: { id: AgentType; label: string; tag: string; desc: string }[] = [
  { id: "react", label: "ReAct Tool Agent", tag: "autonomous", desc: "Reasons in a loop — Thought → Action (tool) → Observation → … → Final Answer. Picks and calls tools on its own." },
  { id: "workflow", label: "Multi-step Workflow", tag: "deterministic", desc: "A fixed pipeline of LLM steps (e.g. plan → draft → critique → finalize), each feeding the next — deterministic & easy to debug." },
];
const TOOL_META: Record<string, { icon: string; label: string }> = {
  calculator: { icon: "🧮", label: "Calculator" },
  datetime: { icon: "🕐", label: "Date & time" },
  web_fetch: { icon: "🌐", label: "Web fetch" },
  knowledge: { icon: "📚", label: "Knowledge base" },
  http_request: { icon: "🔌", label: "HTTP request" },
  human_approval: { icon: "🙋", label: "Human approval" },
  statistics: { icon: "📊", label: "Statistics" },
  unit_convert: { icon: "📐", label: "Unit convert" },
  json_extract: { icon: "🔎", label: "JSON extract" },
  web_search: { icon: "🔍", label: "Web search" },
  wikipedia: { icon: "📖", label: "Wikipedia" },
  arxiv: { icon: "🔬", label: "arXiv" },
  memory: { icon: "🧠", label: "Memory" },
  db_schema: { icon: "🗄", label: "DB schema" },
  db_query: { icon: "🐘", label: "DB query" },
  github: { icon: "🐙", label: "GitHub" },
  mcp: { icon: "🔌", label: "MCP server" },
};

// One-click starter agents — teach where/why agents are used.
const TEMPLATES: { id: string; label: string; tag: string; desc: string; type: AgentType; name: string; goal: string; tools: string[]; knowledge?: string; task: string }[] = [
  { id: "support", label: "Support bot", tag: "customer support", desc: "Answers policy questions from your docs and asks a human before big refunds.", type: "react", name: "Support bot", goal: "You are a customer-support agent. Answer refund & shipping questions grounded in the knowledge base, and use human_approval before authorising any refund over $500.", tools: ["knowledge", "human_approval"], knowledge: "Returns policy: damaged items may be returned within 30 days of delivery for a full refund. Refunds go to the original payment method within 5 business days. Shipping is free over $50, else a $6 flat fee. Gift cards are non-refundable. Refunds above $500 require a manager's approval.", task: "A customer says their $650 order arrived damaged and wants a full refund. What should we do?" },
  { id: "research", label: "Research assistant", tag: "knowledge work", desc: "Reads web pages and gives a concise, grounded answer.", type: "react", name: "Research assistant", goal: "You research questions using web_fetch and reply with a concise, well-structured answer. Cite the page you used.", tools: ["web_fetch", "calculator"], task: "Fetch https://en.wikipedia.org/wiki/Retrieval-augmented_generation and summarise what RAG is in 3 bullet points." },
  { id: "analyst", label: "Data analyst", tag: "analytics", desc: "Answers quantitative questions with the calculator + your data.", type: "react", name: "Data analyst", goal: "You answer quantitative business questions. Use the calculator for arithmetic and the knowledge base for figures.", tools: ["calculator", "knowledge"], knowledge: "Q3 figures: revenue = 240000, cost of goods = 175000, marketing spend = 22000, new customers = 480.", task: "What was the gross margin percentage in Q3, and the marketing cost per new customer?" },
  { id: "api", label: "API agent", tag: "integration", desc: "Calls a public REST API as a tool and reports the result.", type: "react", name: "API agent", goal: "You answer questions by calling public REST APIs with http_request and reading the JSON.", tools: ["http_request", "calculator"], task: "Use http_request on https://api.github.com/repos/vercel/next.js and tell me how many stars the repo has." },
];

// ── chat helper (OpenAI-compatible proxy) ──
async function chatOnce(messages: { role: string; content: string }[], temperature: number, maxTokens: number, providerId?: string, model?: string): Promise<string> {
  const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature, maxTokens, streaming: false, providerId: providerId || undefined, model: model || undefined }) });
  if (!res.ok) { const j = await res.json().catch(() => ({ error: "request failed" })); throw new Error(j.error || "request failed"); }
  return (await res.text()).trim();
}

type TraceItem = { kind: "thought" | "action" | "observation" | "final" | "error"; text: string; tool?: string; state: string };
type ANode = { id: string; type: "trigger" | "agent" | "output" | "model" | "tool"; toolId?: string; icon: string; title: string; sub: string; w: number; h: number; bottom?: string[] };

const CANVAS_W = 980, CANVAS_H = 400;
const DEFAULT_POS: Record<string, { x: number; y: number }> = { trigger: { x: 24, y: 130 }, agent: { x: 306, y: 130 }, output: { x: 772, y: 130 } };
const CopySvg = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

const CheckSvg = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button className="out-copy-btn" onClick={onCopy} title={copied ? "Copied!" : "Copy to clipboard"}>
      {copied ? <CheckSvg /> : <CopySvg />}
    </button>
  );
}

function renderHighlightedText(text: string) {
  if (!text) return null;
  const paragraphs = text.split("\n\n");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {paragraphs.map((p, pIdx) => {
        const lines = p.split("\n");
        return (
          <div key={pIdx} style={{ lineHeight: 1.8, fontSize: 14.5, color: "var(--text)", fontWeight: 500 }}>
            {lines.map((line, lIdx) => {
              const numMatch = line.match(/^(\d+\.\s*)([^:]+)(:?)(.*)$/);
              if (numMatch) {
                return (
                  <div key={lIdx} style={{ margin: "6px 0", display: "flex", alignItems: "flex-start", gap: 10 }}>
                    <span style={{
                      background: "rgba(99,102,241,0.22)",
                      color: "var(--accent-strong)",
                      border: "1.5px solid rgba(99,102,241,0.45)",
                      borderRadius: 7,
                      padding: "2px 9px",
                      fontSize: 13,
                      fontWeight: 800,
                      flexShrink: 0
                    }}>
                      {numMatch[1].trim()}
                    </span>
                    <div>
                      <b style={{ color: "var(--text)", fontWeight: 800, fontSize: 14.5 }}>{numMatch[2]}</b>
                      {numMatch[3] && <span style={{ color: "var(--accent)", fontWeight: 800 }}>: </span>}
                      <span style={{ color: "var(--text)", opacity: 0.95 }}>{numMatch[4]}</span>
                    </div>
                  </div>
                );
              }
              return <div key={lIdx}>{line}</div>;
            })}
          </div>
        );
      })}
    </div>
  );
}

function renderTutorMessage(text: string) {
  // Strip code slashes '// ' from non-code text to keep explanations clean & readable
  const cleanText = text.replace(/^\/\/\s*/gm, "");

  if (cleanText.includes("[BOX_GENAI]") && cleanText.includes("[BOX_AGENT]")) {
    const parts = cleanText.split(/\[BOX_GENAI\]|\[\/BOX_GENAI\]|\[BOX_AGENT\]|\[\/BOX_AGENT\]/);
    const intro = parts[0]?.trim();
    const genAiContent = parts[1]?.trim();
    const agentContent = parts[3]?.trim();
    const footer = parts[4]?.trim();

    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 12, width: "100%" }}>
        {intro && (
          <div style={{ fontWeight: 800, fontSize: 13.5, color: "var(--accent-strong)", borderBottom: "1px solid var(--border)", paddingBottom: 6 }}>
            {intro}
          </div>
        )}

        {/* Side-by-Side Comparison Boxes */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12, margin: "4px 0" }}>
          {/* Box 1: Generative AI */}
          <div style={{
            background: "rgba(99, 102, 241, 0.08)",
            border: "1.5px solid rgba(99, 102, 241, 0.35)",
            borderRadius: 12,
            padding: 14,
            display: "flex",
            flexDirection: "column",
            gap: 8,
            boxShadow: "0 4px 12px rgba(99,102,241,0.08)"
          }}>
            <div style={{ fontWeight: 800, fontSize: 13, color: "#6366f1", borderBottom: "1px solid rgba(99, 102, 241, 0.25)", paddingBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
              {genAiContent?.split("\n")[0]}
            </div>
            <div style={{ fontSize: 12.5, color: "var(--text)", lineHeight: 1.65, whiteSpace: "pre-wrap", fontWeight: 500 }}>
              {genAiContent?.split("\n").slice(1).join("\n").trim()}
            </div>
          </div>

          {/* Box 2: AI Agent */}
          <div style={{
            background: "rgba(168, 85, 247, 0.08)",
            border: "1.5px solid rgba(168, 85, 247, 0.35)",
            borderRadius: 12,
            padding: 14,
            display: "flex",
            flexDirection: "column",
            gap: 8,
            boxShadow: "0 4px 12px rgba(168, 85, 247, 0.08)"
          }}>
            <div style={{ fontWeight: 800, fontSize: 13, color: "#a855f7", borderBottom: "1px solid rgba(168, 85, 247, 0.25)", paddingBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
              {agentContent?.split("\n")[0]}
            </div>
            <div style={{ fontSize: 12.5, color: "var(--text)", lineHeight: 1.65, whiteSpace: "pre-wrap", fontWeight: 500 }}>
              {agentContent?.split("\n").slice(1).join("\n").trim()}
            </div>
          </div>
        </div>

        {footer && (
          <div style={{
            background: "var(--panel)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: "10px 14px",
            fontSize: 12.5,
            color: "var(--text)",
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
            fontWeight: 500
          }}>
            {footer}
          </div>
        )}
      </div>
    );
  }

  return <div style={{ whiteSpace: "pre-wrap", fontSize: 13.5, color: "var(--text)", lineHeight: 1.7, fontWeight: 500 }}>{cleanText}</div>;
}

export default function AgentLab() {
  const [step, setStep] = useState<Step>("type");
  const [selectedTopic, setSelectedTopic] = useState<LearnTopic | null>(null);
  const [quizOption, setQuizOption] = useState<number | null>(null);
  const [tutorMessages, setTutorMessages] = useState<{ sender: "user" | "bot"; text: string }[]>([
    {
      sender: "bot",
      text: "🤖 Welcome to the AI Agent Tutor Bot!\nAsk me any question about What is an AI Agent, Generative AI vs Agents, Chunks, Embeddings, APIs, LLMs, or Implementation steps."
    }
  ]);
  const [tutorInput, setTutorInput] = useState("");
  const [tutorialActive, setTutorialActive] = useState(false);
  const [tutorialStepIdx, setTutorialStepIdx] = useState(0);

  const askTutor = (questionText: string) => {
    if (!questionText.trim()) return;
    const userQ = questionText.trim();
    setTutorMessages((prev) => [...prev, { sender: "user", text: userQ }]);
    setTutorInput("");

    // Guardrail Check
    const lower = userQ.toLowerCase();
    const isAgentRelated = /agent|genai|generative|llm|model|chunk|embedding|vector|rag|api|tool|react|workflow|prompt|hitl|human|step|implementation|perception|action|reason/i.test(lower);

    if (!isAgentRelated) {
      setTimeout(() => {
        setTutorMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: "⚠️ Guardrail Notice\nI am your dedicated AI Agent Tutor Bot. I can only assist with questions about AI Agent Fundamentals, Architecture, Chunks, Embeddings, RAG, and Implementation.\nPlease ask a question related to AI Agents!"
          }
        ]);
      }, 300);
      return;
    }

    // Knowledge matching — Comprehensive & Detailed Explanations
    let botReply = "";

    if (lower.includes("difference") || lower.includes("genai") || lower.includes("vs")) {
      botReply = `⚖️ Generative AI vs. AI Agent — Side-by-Side Comparison:

[BOX_GENAI]
🤖 Generative AI (Text Generator)
• Core Role: Foundation LLM (GPT-4, Claude, Llama) predicting text/code.
• Execution Mode: Passive 1-shot turn (Prompt → Response string).
• Real-World Example: You ask ChatGPT "Write a python script to pull stock prices" → Outputs code text, but cannot execute it or email results.
• System Scope: Sandboxed in passive text generation without native tools or execution runtimes.
[/BOX_GENAI]

[BOX_AGENT]
⚡ AI Agent (Autonomous System)
• Core Role: Cognitive Brain + Tools + Memory + Perception Feedback Loop.
• Execution Mode: Active multi-step goal loop (Goal → Reasoning → Tool Action → Observation).
• Real-World Example: You ask agent "Fetch TSLA price, compute 10 shares value, email report to boss" → Calls stock_api ($210.50), calculator ($2,105.00), and email_api automatically!
• System Scope: Autonomously executes actions in the real world across databases, APIs, and systems.
[/BOX_AGENT]

💡 Key Difference: Generative AI is the passive writing brain, while an AI Agent is the active goal-oriented system equipped with hands, memory, and tools.`;
    } else if (lower.includes("chunk") || lower.includes("embedding") || lower.includes("rag")) {
      botReply = `📚 Document Chunks & Vector Embeddings in RAG

1. Why Chunking & Vector Embeddings Are Essential:
- Overcoming Context Window Limits: A 500-page corporate manual (1,000,000 tokens) cannot fit into an LLM context window.
- Preventing Hallucinations: Grounding LLM responses in exact retrieved document paragraphs ensures 100% factual accuracy.

2. Step-by-Step RAG Pipeline Execution Flow:
- Step 1 (Text Chunking): Raw document text is sliced into 250–500 token chunks with 10-20% overlapping boundaries to preserve sentence context.
- Step 2 (Vector Embedding): An embedding model (text-embedding-3-small) encodes each chunk into a 1536-dimensional float vector array representing its semantic meaning.
- Step 3 (Vector DB Storage): Embeddings and text metadata are indexed in a vector database (Pinecone, ChromaDB, FAISS).
- Step 4 (Cosine Similarity Search): User prompt is vectorized; Vector DB measures spatial Cosine Distance to retrieve top K (e.g. 3) closest chunks.
- Step 5 (Augmented Generation): Retrieved chunks are injected into LLM context window to generate a grounded answer with source citations.

3. Concrete Code Example (Python):
# 1. Chunk text
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_text(raw_pdf_text)

# 2. Vectorize & Search
vector_db = Chroma.from_texts(chunks, embedding=OpenAIEmbeddings())
top_chunks = vector_db.similarity_search("What is refund policy?", k=3)`;
    } else if (lower.includes("api") || lower.includes("why model") || lower.includes("llm")) {
      botReply = `⚡ Why LLMs & APIs Are Used Together

1. The Synergistic Architecture (Brain + Hands):
- The LLM (Cognitive Brain): Handles natural language understanding, intent classification, entity extraction, planning, and synthesis. However, an LLM alone is isolated in a text sandbox unable to fetch live data or modify external databases.
- REST APIs (External Hands & Feet): Expose real software functions (HTTP requests, SQL queries, calculators, payment gateways).

2. End-to-End Execution Trace Example:
- User Prompt: "Check Apple stock price and calculate value of 10 shares."
- Step 1 (LLM Brain Reasoning): LLM parses prompt -> decides tool call needed: stock_api | params: { ticker: 'AAPL' }.
- Step 2 (API Execution): Application runtime calls Financial REST API: GET /api/v1/stock/AAPL -> receives JSON: { price: 220.00 }.
- Step 3 (Observation Feedback): API result { price: 220.00 } is fed back to LLM context as an Observation.
- Step 4 (LLM Second Pass): LLM sees AAPL=$220 -> decides next tool call: calculator | params: 10 * 220.
- Step 5 (API Execution 2): Calculator returns Observation: 2200.
- Step 6 (Final Response): LLM synthesizes: "Apple (AAPL) is $220.00/share. 10 shares equal $2,200.00."

3. Industry Applications:
- ChatGPT Custom GPTs (Connecting OpenAI reasoning to Zapier/KAYAK REST APIs).
- Automated Customer Care Bots (Connecting LLMs to Salesforce CRM & Stripe payment APIs).`;
    } else if (lower.includes("understand") || lower.includes("message") || lower.includes("flow")) {
      botReply = `🔄 How AI Agents Process Messages & Execute

1. The 6-Stage Deep Lifecycle Sequence:
- Stage 1 (Raw Ingestion): User submits natural language query (e.g. "Calculate 15% tip on $85 bill").
- Stage 2 (BPE Tokenization): Text is tokenized into numerical sub-word integer IDs using tiktoken encoder.
- Stage 3 (Context Window Assembly): System Prompt rules + Conversation History + Tool Schemas + User Tokens are concatenated into an assembled context tensor array.
- Stage 4 (Transformer Self-Attention): Multi-head self-attention layers compute Q, K, V token weights to parse intent and extract parameters.
- Stage 5 (Tool Dispatch / Text Output): LLM evaluates whether to emit a direct text string OR a structured tool request payload (e.g. Action: calculator | Input: 85 * 0.15).
- Stage 6 (Observation Feedback & Response): Backend executes tool payload, appends Observation back to context, and LLM synthesizes final answer.`;
    } else if (lower.includes("react") || lower.includes("loop")) {
      botReply = `🤖 ReAct (Reasoning + Acting) Loop Architecture

1. Core Concept (Yao et al., 2023):
- ReAct is an autonomous prompting and execution pattern where an LLM alternates between explicit internal reasoning (Thought), tool invocation (Action), and tool result parsing (Observation) in a loop until reaching a confident Final Answer.

2. Step-by-Step ReAct Trace Example:
- Goal: "What is the weather in Paris and what should I wear?"
- Turn 1:
  • Thought: I need to check the current weather in Paris first.
  • Action: get_weather | Action Input: { city: "Paris" }
  • Observation: { temp: "12C", condition: "Rainy" }
- Turn 2:
  • Thought: It is 12C and rainy in Paris. I should recommend warm clothes and an umbrella.
  • Final Answer: Current weather in Paris is 12C and rainy. Recommended outfit: warm jacket, waterproof boots, and an umbrella!`;
    } else if (lower.includes("step") || lower.includes("build") || lower.includes("implement")) {
      botReply = `🚀 5 General Steps to Implement an AI Agent

1. Step 1 — Persona & System Instructions:
- Establish identity, role, negative constraints, and output formatting rules in system prompt.

2. Step 2 — Tools & Function Schemas:
- Expose external code (APIs, calculators, SQL connectors) as JSON parameter schemas.

3. Step 3 — Cognitive Control Loop Architecture:
- Select ReAct loop (for open-ended autonomous tasks) or Deterministic Workflow DAG (for fixed step pipelines).

4. Step 4 — Vector RAG Knowledge Store Integration:
- Ingest private PDFs into a vector database (Pinecone/ChromaDB) for semantic context retrieval.

5. Step 5 — Security Guardrails & Human-in-the-Loop (HITL):
- Set iteration caps (max 6 loops) and require human admin approval for high-risk actions (refunds, database deletes).`;
    } else {
      botReply = `🧠 AI Agent Core Concept Overview

1. Definition:
An AI Agent is a goal-oriented autonomous software system that uses a Large Language Model (LLM) as its central cognitive brain. It receives environmental inputs, plans multi-step actions, invokes external tools (APIs, calculators, databases), reads observation feedback, and continuously loops until fulfilling the user's objective.

2. Key Components:
- Cognitive Brain (LLM): Natural language understanding, reasoning, and planning.
- Environment Tools (APIs): Real-world code execution and data access.
- Memory: Conversation history and vector database context.
- Feedback Loop: Perception-Action loop evaluating tool outputs.

Explore any connected node on the Concept Network Tree above for detailed visual flowcharts, code blueprints, and industry case studies!`;
    }

    setTimeout(() => {
      setTutorMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
    }, 400);
  };
  const [agentType, setAgentType] = useState<AgentType>("react");
  const [runtime, setRuntime] = useState<"browser" | "nat">("browser");
  const [buildMode, setBuildMode] = useState<"visual" | "manual" | "prompt">("visual");

  // providers / models
  const [providers, setProviders] = useState<Provider[]>([]);
  const [providerId, setProviderId] = useState("");
  const [modelList, setModelList] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [provKnown, setProvKnown] = useState(false);
  const [msg, setMsg] = useState("");

  // shared / react config
  const [name, setName] = useState("Support agent");
  const [description, setDescription] = useState("Answers questions and uses tools when it needs facts.");
  const [goal, setGoal] = useState("You are a helpful support agent. Use the connected tools when you need fresh facts, and cite what you use.");
  const [temperature, setTemperature] = useState(0.4);
  const [maxTokens, setMaxTokens] = useState(600);
  const [maxIters, setMaxIters] = useState(6);
  const [enabledTools, setEnabledTools] = useState<Set<string>>(new Set(["calculator", "datetime", "knowledge"]));
  const [placedTools, setPlacedTools] = useState<Set<string>>(new Set(["calculator", "datetime", "knowledge"]));
  const [knowledgeText, setKnowledgeText] = useState("Returns policy: damaged items may be returned within 30 days of delivery for a full refund. Shipping is free on orders over $50, otherwise a flat $6 fee applies. Gift cards are non-refundable.");
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // workflow config
  const [steps, setSteps] = useState<{ id: string; name: string; instruction: string }[]>([
    { id: "s1", name: "Plan", instruction: "Break the user's request into a short bullet plan." },
    { id: "s2", name: "Draft", instruction: "Write a first draft that follows the plan." },
    { id: "s3", name: "Critique", instruction: "List concrete weaknesses in the draft." },
    { id: "s4", name: "Finalize", instruction: "Rewrite the draft addressing every critique. Output only the final result." },
  ]);

  // from-prompt
  const [describe, setDescribe] = useState("A finance helper that can do calculations and look things up online.");
  const [generating, setGenerating] = useState(false);

  // node canvas
  const [nodePos, setNodePos] = useState<Record<string, { x: number; y: number }>>({});
  const [aSel, setASel] = useState("agent");
  const [addOpen, setAddOpen] = useState(false);
  const [nodeStatus, setNodeStatus] = useState<Record<string, string>>({});
  const [fullscreen, setFullscreen] = useState(false);
  const [connectFrom, setConnectFrom] = useState<{ id: string; kind: "tool" | "agent" } | null>(null);
  const [connectXY, setConnectXY] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [saved, setSaved] = useState(false);
  const [savedProjectId, setSavedProjectId] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [published, setPublished] = useState(false);
  const [savedAgents, setSavedAgents] = useState<{ id: string; name: string; config: Record<string, unknown>; published?: boolean }[]>([]);
  const [loadOpen, setLoadOpen] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  // run
  const [task, setTask] = useState("What is 15% of 240, and how many days until 2026-12-25?");
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [finalOut, setFinalOut] = useState("");
  const [wfOutputs, setWfOutputs] = useState<{ name: string; text: string; state: string }[]>([]);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [pendingApproval, setPendingApproval] = useState<{ q: string; resolve: (v: string) => void } | null>(null);
  const [metrics, setMetrics] = useState<{ calls: number; tools: number; ms: number; tokens: number } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  function applyTemplate(t: typeof TEMPLATES[number]) {
    setAgentType(t.type); setName(t.label); setDescription(t.desc); setGoal(t.goal);
    setEnabledTools(new Set(t.tools)); setPlacedTools(new Set(t.tools)); if (t.knowledge) setKnowledgeText(t.knowledge);
    setTask(t.task); setNodePos({}); setNodeStatus({}); setASel("agent"); setBuildMode("visual"); setStep("build");
  }

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((j) => {
      setProvKnown(true);
      setProviders(j.providers || []);
      if (j.providers?.length) { setProviderId(j.providerId || j.providers[0].id); setModelList(j.models || []); setModel(j.default || (j.models && j.models[0]) || ""); }
    }).catch(() => setProvKnown(true));
  }, []);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [trace, wfOutputs]);
  // Delete / Backspace removes the selected tool node (or clicked wire's tool).
  useEffect(() => {
    if (step !== "build" || agentType !== "react" || buildMode !== "visual") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      const el = document.activeElement as HTMLElement | null;
      if (el && ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      if (aSel.startsWith("tool:")) { e.preventDefault(); const id = aSel.slice(5); setPlacedTools((s) => { const n = new Set(s); n.delete(id); return n; }); setEnabledTools((s) => { const n = new Set(s); n.delete(id); return n; }); setASel("agent"); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, agentType, buildMode, aSel]);

  const hasProvider = providers.length > 0;
  const toolList = AGENT_TOOLS.filter((t) => enabledTools.has(t.id));
  const togglePlaced = (id: string) => { setNodePos({}); setPlacedTools((s) => { const n = new Set(s); if (n.has(id)) { n.delete(id); setEnabledTools((e) => { const m = new Set(e); m.delete(id); return m; }); } else { n.add(id); setEnabledTools((e) => new Set(e).add(id)); } return n; }); };
  const connectTool = (id: string) => { setNodePos({}); setPlacedTools((s) => new Set(s).add(id)); setEnabledTools((s) => new Set(s).add(id)); };
  const disconnectTool = (id: string) => { setNodePos({}); setEnabledTools((s) => { const n = new Set(s); n.delete(id); return n; }); };
  const removeToolNode = (id: string) => { setNodePos({}); setPlacedTools((s) => { const n = new Set(s); n.delete(id); return n; }); setEnabledTools((s) => { const n = new Set(s); n.delete(id); return n; }); };
  const toggleManual = (id: string) => (enabledTools.has(id) ? disconnectTool(id) : connectTool(id));
  function loadModels(pid: string) {
    setProviderId(pid);
    fetch(`/api/models?providerId=${pid}`).then((r) => r.json()).then((j) => { setModelList(j.models || []); setModel(j.default || (j.models && j.models[0]) || ""); }).catch(() => {});
  }
  const providerLabel = providers.find((p) => p.id === providerId)?.label || providers.find((p) => p.id === providerId)?.provider || "provider";

  // ── knowledge upload ──
  async function onKnowledgeFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]; if (!f) return;
    setUploading(true); setMsg("");
    const ext = (f.name.split(".").pop() || "txt").toLowerCase();
    try {
      if (["pdf", "docx", "doc", "xlsx", "xls", "xlsm"].includes(ext)) {
        const fd = new FormData(); fd.append("file", f);
        const r = await fetch("/api/rag/extract", { method: "POST", body: fd });
        const j = await r.json(); if (!r.ok) throw new Error(j.error || "parse failed");
        setKnowledgeText(j.text);
      } else setKnowledgeText(await f.text());
    } catch (err) { setMsg(`Could not read ${f.name}: ${(err as Error).message}`); }
    setUploading(false); e.target.value = "";
  }

  // ── node canvas model ──
  const placedOrder = [...placedTools];
  const subNodes = ["model", ...placedOrder.map((tid) => "tool:" + tid)];

  const getSubDefaultPos = (subId: string) => {
    const idx = subNodes.indexOf(subId);
    if (idx === -1) return { x: 150, y: 300 };
    const total = subNodes.length;
    const nodeW = 160;
    const gap = 24;
    const totalW = total * nodeW + (total - 1) * gap;
    const startX = Math.max(20, 406 - totalW / 2);
    return { x: Math.round(startX + idx * (nodeW + gap)), y: 300 };
  };

  const getPos = (id: string) => nodePos[id] || DEFAULT_POS[id] || getSubDefaultPos(id);

  const nodes: ANode[] = [
    { id: "trigger", type: "trigger", icon: "💬", title: "User input", sub: "Trigger", w: 150, h: 54 },
    { id: "agent", type: "agent", icon: "🤖", title: name || "AI Agent", sub: "ReAct agent", w: 200, h: 54 },
    { id: "output", type: "output", icon: "✅", title: "Final answer", sub: "Output", w: 150, h: 54 },
    { id: "model", type: "model", icon: "⚙️", title: (model || providerLabel).slice(0, 18), sub: "Model", w: 160, h: 54 },
    ...placedOrder.map((tid) => ({ id: "tool:" + tid, type: "tool" as const, toolId: tid, icon: TOOL_META[tid]?.icon ?? "🔧", title: TOOL_META[tid]?.label ?? tid, sub: tid === "knowledge" ? "Knowledge" : "Tool", w: 160, h: 54 })),
  ];

  const agentNode = nodes.find((n) => n.id === "agent")!;

  function portPos(n: ANode, which: "in" | "out" | "top" | string | number): [number, number] {
    const p = getPos(n.id);
    if (which === "in") return [p.x, p.y + n.h / 2];
    if (which === "out") return [p.x + n.w, p.y + n.h / 2];
    if (which === "top") return [p.x + n.w / 2, p.y];
    const idx = typeof which === "number" ? which : subNodes.indexOf(String(which));
    const total = Math.max(1, subNodes.length);
    const safeIdx = idx >= 0 ? idx : 0;
    const portX = p.x + (n.w * (safeIdx + 1)) / (total + 1);
    return [portX, p.y + n.h];
  }

  const wires = [
    { from: "trigger", to: "agent", kind: "main" },
    { from: "agent", to: "output", kind: "main" },
    { from: "model", to: "agent", kind: "sub" },
    ...[...enabledTools].filter((tid) => placedTools.has(tid)).map((tid) => ({ from: "tool:" + tid, to: "agent", kind: "sub" })),
  ];

  function wirePath(w: { from: string; to: string; kind: string }): string {
    const a = nodes.find((n) => n.id === w.from), b = nodes.find((n) => n.id === w.to);
    if (!a || !b) return "";
    if (w.kind === "main") {
      const [x1, y1] = portPos(a, "out");
      const [x2, y2] = portPos(b, "in");
      const dx = Math.max(40, (x2 - x1) / 2);
      return `M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
    }
    const [sx, sy] = portPos(a, "top");
    const [bx, by] = portPos(agentNode, w.from);
    const dy = Math.max(30, Math.abs(sy - by) / 2);
    return `M${bx} ${by} C${bx} ${by + dy}, ${sx} ${sy - dy}, ${sx} ${sy}`;
  }

  function onNodeDown(e: React.PointerEvent, id: string) {
    const start = getPos(id); const sx = e.clientX, sy = e.clientY; let moved = false;
    const mv = (ev: PointerEvent) => { const dx = ev.clientX - sx, dy = ev.clientY - sy; if (Math.abs(dx) + Math.abs(dy) > 4) moved = true; setNodePos((p) => ({ ...p, [id]: { x: Math.max(0, start.x + dx), y: Math.max(0, start.y + dy) } })); };
    const up = () => { document.removeEventListener("pointermove", mv); document.removeEventListener("pointerup", up); if (!moved) setASel(id); };
    document.addEventListener("pointermove", mv); document.addEventListener("pointerup", up); e.preventDefault();
  }

  // Drag a port to another node to wire it up (n8n-style start→end connection).
  function portDown(e: React.PointerEvent, id: string, kind: "tool" | "agent") {
    e.stopPropagation(); e.preventDefault();
    const canvas = canvasRef.current; if (!canvas) return;
    const rel = (cx: number, cy: number) => { const r = canvas.getBoundingClientRect(); return { x: cx - r.left + canvas.scrollLeft, y: cy - r.top + canvas.scrollTop }; };
    setConnectFrom({ id, kind }); setConnectXY(rel(e.clientX, e.clientY));
    const mv = (ev: PointerEvent) => setConnectXY(rel(ev.clientX, ev.clientY));
    const up = (ev: PointerEvent) => {
      document.removeEventListener("pointermove", mv); document.removeEventListener("pointerup", up);
      const pt = rel(ev.clientX, ev.clientY);
      const hit = nodes.find((n) => { const p = getPos(n.id); return pt.x >= p.x && pt.x <= p.x + n.w && pt.y >= p.y && pt.y <= p.y + n.h; });
      if (hit) {
        if (kind === "tool" && hit.id === "agent") connectTool(id.replace("tool:", ""));
        else if (kind === "agent" && hit.type === "tool") connectTool(hit.toolId!);
      }
      setConnectFrom(null);
    };
    document.addEventListener("pointermove", mv); document.addEventListener("pointerup", up);
  }

  function tempWirePath(): string {
    if (!connectFrom) return "";
    let src: [number, number];
    if (connectFrom.kind === "agent") src = portPos(agentNode, 0);
    else { const tn = nodes.find((n) => n.id === connectFrom.id); if (!tn) return ""; src = portPos(tn, "top"); }
    return `M${src[0]} ${src[1]} C${src[0]} ${src[1] + 40}, ${connectXY.x} ${connectXY.y - 40}, ${connectXY.x} ${connectXY.y}`;
  }

  const flowCanvas = () => (
    <div className="acanvas" ref={canvasRef} onClick={() => setAddOpen(false)}>
      <svg className="wires2" width="100%" height={CANVAS_H}>
        {wires.map((w, i) => {
          const fromSt = nodeStatus[w.from] || "";
          const toSt = nodeStatus[w.to] || "";
          const isRunning = fromSt === "running" || toSt === "running";
          const isDone = (fromSt === "done" || fromSt === "running") && (toSt === "done" || toSt === "running");
          const sel = w.kind === "sub" && aSel === w.from;
          const wireClass = `${w.kind === "sub" ? "sub" : ""} ${isRunning ? "active-running" : isDone ? "active-done" : ""} ${sel ? "selw" : ""}`;
          return <path key={i} className={wireClass} d={wirePath(w)} />;
        })}
        {wires.filter((w) => w.kind === "sub").map((w, i) => <path key={"hit" + i} className="wire-hit" d={wirePath(w)} onClick={(e) => { e.stopPropagation(); setASel(w.from); }} />)}
        {connectFrom && <path className="sub selw" d={tempWirePath()} />}
      </svg>
      {nodes.map((n) => { const pos = getPos(n.id); const st = nodeStatus[n.id] || ""; const unwired = n.type === "tool" && !enabledTools.has(n.toolId!); return (
        <div key={n.id} className={`anode type-${n.type} ${aSel === n.id ? "sel" : ""} ${st} ${unwired ? "unwired" : ""}`} style={{ left: pos.x, top: pos.y, width: n.w }} onPointerDown={(e) => onNodeDown(e, n.id)}>
          <div className="ah"><span className="aic">{n.icon}</span><div><div className="atitle">{n.title}</div><div className="asub">{unwired ? "drag ↑ to connect" : n.sub}</div></div><span className="abadge" /></div>
          {n.type === "tool" && <span className="aport ap-top" title="drag to the Agent to connect" onPointerDown={(e) => portDown(e, n.id, "tool")} />}
          {n.type === "agent" && subNodes.map((subId, idx) => {
            const px = (n.w * (idx + 1)) / (subNodes.length + 1);
            return (
              <span
                key={subId}
                className="aport ap-agent-bottom"
                style={{ left: px, bottom: -5 }}
                title="drag to a tool to connect"
                onPointerDown={(e) => portDown(e, "agent", "agent")}
              />
            );
          })}
          {n.type === "tool" && <button className="anode-x" title="Remove node" onPointerDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); removeToolNode(n.toolId!); if (aSel === n.id) setASel("agent"); }}>×</button>}
        </div>
      ); })}
    </div>
  );

  // ── from-prompt generation ──
  async function generateFromPrompt() {
    setGenerating(true); setMsg("");
    try {
      const react = agentType === "react";
      const sys = react ? "Design a ReAct tool agent from the user's description." : "Design a multi-step LLM workflow from the user's description.";
      const schema = react
        ? `{"name": string, "goal": string, "tools": string[] (subset of ["calculator","datetime","web_fetch","knowledge","http_request","human_approval"]), "maxIters": number}`
        : `{"name": string, "steps": [{"name": string, "instruction": string}] (3-5)}`;
      const out = await chatOnce([{ role: "system", content: `${sys} Reply with ONLY minified JSON matching: ${schema}. No prose, no code fences.` }, { role: "user", content: describe }], 0.3, 500, providerId, model);
      const cfg = JSON.parse(out.slice(out.indexOf("{"), out.lastIndexOf("}") + 1));
      if (cfg.name) setName(String(cfg.name));
      if (react) {
        if (cfg.goal) setGoal(String(cfg.goal));
        if (Array.isArray(cfg.tools)) { const picked = new Set<string>(cfg.tools.filter((t: string) => AGENT_TOOLS.some((x) => x.id === t))); setEnabledTools(picked); setPlacedTools(new Set(picked)); }
        if (cfg.maxIters) setMaxIters(Math.max(1, Math.min(10, Number(cfg.maxIters) || 6)));
        setBuildMode("visual");
      } else if (Array.isArray(cfg.steps)) setSteps(cfg.steps.slice(0, 6).map((s: { name?: string; instruction?: string }, i: number) => ({ id: `s${i + 1}`, name: String(s.name || `Step ${i + 1}`), instruction: String(s.instruction || "") })));
    } catch (e) { setMsg("Could not generate config: " + (e as Error).message); }
    setGenerating(false);
  }

  // ── RUN ──
  async function runReact() {
    setRunning(true); setTrace([]); setFinalOut(""); setMsg(""); setMetrics(null); setPendingApproval(null);
    setNodeStatus({ trigger: "done", model: "done", agent: "running" });
    const ctx: ToolCtx = { requestApproval: (q) => new Promise<string>((resolve) => setPendingApproval({ q, resolve })) };
    if (enabledTools.has("knowledge")) { const kb = buildKnowledge(knowledgeText); if (kb) { ctx.knowledgeIndex = kb.index as RagIndex; ctx.knowledgeChunks = kb.chunks; } }
    const tools: AgentTool[] = toolList;
    const messages: { role: string; content: string }[] = [{ role: "system", content: reactSystemPrompt(tools, goal) }, { role: "user", content: task }];
    const push = (t: TraceItem) => setTrace((tr) => [...tr, t]);
    const est = (s: string) => Math.round(s.length / 4);
    let calls = 0, toolsUsed = 0, tokens = 0; const t0 = performance.now();
    const toolCounts: Record<string, number> = {};
    let iterations = 0, outcome = "max_iters", errorMsg = "";
    try {
      for (let iter = 0; iter < maxIters; iter++) {
        iterations = iter + 1;
        push({ kind: "thought", text: "thinking…", state: "active" });
        setNodeStatus((s) => ({ ...s, agent: "running" }));
        tokens += messages.reduce((a, m) => a + est(m.content), 0); calls++;
        const resp = await chatOnce(messages, temperature, Math.min(maxTokens, 500), providerId, model);
        tokens += est(resp);
        const p = parseReAct(resp);
        setTrace((tr) => { const c = [...tr]; c[c.length - 1] = { kind: "thought", text: p.thought || "(reasoning)", state: "done" }; return c; });
        if (p.final || (!p.action && !p.final)) { const ans = formatFinalAnswer(p.final || resp); setFinalOut(ans); push({ kind: "final", text: ans, state: "done" }); setNodeStatus((s) => ({ ...s, agent: "done", output: "done" })); outcome = "success"; break; }
        const tool = tools.find((t) => t.name.toLowerCase() === (p.action || "").toLowerCase());
        push({ kind: "action", text: p.input || "", tool: p.action, state: "active" });
        if (tool) { toolsUsed++; toolCounts[tool.name] = (toolCounts[tool.name] || 0) + 1; setNodeStatus((s) => ({ ...s, ["tool:" + tool.id]: "running" })); }
        const obs = tool ? await tool.run(p.input || "", ctx) : `Unknown tool "${p.action}". Available: ${tools.map((t) => t.name).join(", ")}.`;
        if (tool) setNodeStatus((s) => ({ ...s, ["tool:" + tool.id]: "done" }));
        setTrace((tr) => { const c = [...tr]; c[c.length - 1] = { ...c[c.length - 1], state: "done" }; return c; });
        push({ kind: "observation", text: obs, state: "done" });
        messages.push({ role: "assistant", content: resp }); messages.push({ role: "user", content: `Observation: ${obs}` });
        if (iter === maxIters - 1) { push({ kind: "error", text: `Reached the ${maxIters}-step limit without a final answer.`, state: "done" }); setNodeStatus((s) => ({ ...s, agent: "done", output: "done" })); }
      }
    } catch (e) { outcome = "error"; errorMsg = (e as Error).message; push({ kind: "error", text: (e as Error).message, state: "done" }); }
    const ms = Math.round(performance.now() - t0);
    setMetrics({ calls, tools: toolsUsed, ms, tokens });
    logAgentRun({
      agentName: name, agentType: "react", runtime: "browser", provider: providerLabel, model,
      iterations, toolCalls: Object.entries(toolCounts).map(([tool, count]) => ({ tool, count })),
      toolCallCount: toolsUsed, totalTokens: tokens, latencyMs: ms, outcome, errorMsg,
    });
    setPendingApproval(null); setRunning(false);
  }
  // Fire-and-forget: persist a run summary for the agent analytics dashboard.
  function logAgentRun(payload: Record<string, unknown>) {
    fetch("/api/agent/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) }).catch(() => {});
  }
  async function runWorkflow() {
    setRunning(true); setMsg(""); setFinalOut(""); setWfOutputs(steps.map((s) => ({ name: s.name, text: "", state: "" })));
    let prev = task; let outcome = "success", errorMsg = "", tokens = Math.round(task.length / 4); const t0 = performance.now();
    try {
      for (let k = 0; k < steps.length; k++) {
        setWfOutputs((o) => { const n = [...o]; n[k] = { ...n[k], state: "active" }; return n; });
        const messages = [{ role: "system", content: `You are executing step "${steps[k].name}" of a workflow. ${steps[k].instruction}` }, { role: "user", content: `Original request: ${task}\n\nPrevious step output:\n${prev}` }];
        const res = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ messages, temperature, providerId: providerId || undefined, model: model || undefined }) });
        if (!res.ok || !res.body) { const j = await res.json().catch(() => ({ error: "failed" })); throw new Error(j.error || "failed"); }
        const reader = res.body.getReader(); const dec = new TextDecoder(); let acc = "";
        for (; ;) { const { done, value } = await reader.read(); if (done) break; acc += dec.decode(value, { stream: true }); setWfOutputs((o) => { const n = [...o]; n[k] = { ...n[k], text: acc }; return n; }); }
        setWfOutputs((o) => { const n = [...o]; n[k] = { ...n[k], state: "done" }; return n; });
        prev = acc; tokens += Math.round(acc.length / 4);
      }
      setFinalOut(prev);
    } catch (e) { outcome = "error"; errorMsg = (e as Error).message; setMsg("Workflow error: " + errorMsg); }
    logAgentRun({ agentName: name, agentType: "workflow", runtime: "browser", provider: providerLabel, model, iterations: steps.length, toolCalls: [], toolCallCount: 0, totalTokens: tokens, latencyMs: Math.round(performance.now() - t0), outcome, errorMsg });
    setRunning(false);
  }
  function startRun() { setStep("run"); if (agentType === "react") runReact(); else runWorkflow(); }
  // Linear workflow node canvas (input → steps → output), lights up as it runs.
  const wfCanvas = () => {
    const list = [
      { id: "in", icon: "💬", title: "Input", sub: "Trigger", st: running || wfOutputs.length ? "done" : "" },
      ...steps.map((s, i) => ({ id: s.id, icon: "◆", title: s.name, sub: `Step ${i + 1}`, st: wfOutputs[i]?.state || "" })),
      { id: "out", icon: "✅", title: "Final output", sub: "Output", st: finalOut && !running ? "done" : "" },
    ];
    const W = 150, H = 60, GAP = 44, y = 100;
    const pos = (i: number) => ({ x: 20 + i * (W + GAP), y });
    const cw = Math.max(CANVAS_W, 20 + list.length * (W + GAP));
    return (
      <div className="acanvas" style={{ height: 220 }}>
        <svg className="wires2" width={cw} height={220}>
          {list.slice(0, -1).map((_, i) => { const a = pos(i), b = pos(i + 1); const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2; const dx = Math.max(30, (x2 - x1) / 2); const act = ["running", "done"].includes(list[i].st); return <path key={i} className={act ? "active" : ""} d={`M${x1} ${y1} C${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`} />; })}
        </svg>
        {list.map((n, i) => { const p = pos(i); return (
          <div key={n.id} className={`anode ${n.st}`} style={{ left: p.x, top: p.y, width: W }}>
            <div className="ah"><span className="aic">{n.icon}</span><div><div className="atitle">{n.title}</div><div className="asub">{n.sub}</div></div><span className="abadge" /></div>
          </div>
        ); })}
      </div>
    );
  };

  // ── code export ──
  function buildCode(): string {
    const q3 = (s: string) => s.replace(/\\/g, "\\\\").replace(/"""/g, '\\"\\"\\"');
    const preamble = `import os
from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = ${JSON.stringify(model || "your-model")}`;
    if (agentType === "workflow") {
      const stepsPy = steps.map((s) => `    (${JSON.stringify(s.name)}, ${JSON.stringify(s.instruction)}),`).join("\n");
      return `# AI Workbench · workflow "${name}"  (pip install openai)
${preamble}

STEPS = [
${stepsPy}
]

def run(task):
    prev = task
    for step_name, instruction in STEPS:
        msgs = [
            {"role": "system", "content": f"You are executing step '{step_name}'. {instruction}"},
            {"role": "user", "content": f"Original request: {task}\\n\\nPrevious step output:\\n{prev}"},
        ]
        prev = client.chat.completions.create(model=MODEL, messages=msgs, temperature=${temperature}).choices[0].message.content
        print(f"--- {step_name} ---\\n{prev}\\n")
    return prev

if __name__ == "__main__":
    print(run(${JSON.stringify(task)}))`;
    }
    const tls = toolList;
    const imports = ["import re"]; const defs: string[] = []; const reg: string[] = [];
    if (tls.some((t) => t.id === "calculator")) { imports.push("import math"); reg.push(`"calculator": tool_calculator`); defs.push(`def tool_calculator(x):
    try:
        return str(eval(x, {"__builtins__": {}}, {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}))
    except Exception as e:
        return f"Error: {e}"`); }
    if (tls.some((t) => t.id === "datetime")) { imports.push("import datetime as _dt"); reg.push(`"datetime": tool_datetime`); defs.push(`def tool_datetime(x):
    m = re.search(r"(\\d{4}-\\d{1,2}-\\d{1,2})", x or "")
    if m:
        delta = (_dt.date.fromisoformat(m.group(1)) - _dt.date.today()).days
        return f"{m.group(1)} is {delta} day(s) from today."
    return f"Now: {_dt.datetime.now().isoformat()}"`); }
    if (tls.some((t) => t.id === "web_fetch")) { imports.push("import requests"); reg.push(`"web_fetch": tool_web_fetch`); defs.push(`def tool_web_fetch(x):
    try:
        html = requests.get(x.strip(), timeout=12).text
        return re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", html))[:1200]
    except Exception as e:
        return f"Error: {e}"`); }
    if (tls.some((t) => t.id === "knowledge")) { reg.push(`"knowledge": tool_knowledge`); defs.push(`KNOWLEDGE = """${q3(knowledgeText)}"""

def tool_knowledge(x):
    sents = re.split(r"(?<=[.!?])\\s+", KNOWLEDGE)
    q = set(re.findall(r"[a-z0-9]+", x.lower()))
    ranked = sorted(sents, key=lambda s: len(q & set(re.findall(r"[a-z0-9]+", s.lower()))), reverse=True)
    return " ".join(ranked[:3])[:1200]`); }
    if (tls.some((t) => t.id === "http_request")) { if (!imports.includes("import requests")) imports.push("import requests"); imports.push("import json as _json"); reg.push(`"http_request": tool_http_request`); defs.push(`def tool_http_request(x):
    x = x.strip()
    try:
        if x.startswith("{"):
            spec = _json.loads(x)
            r = requests.request(spec.get("method", "GET").upper(), spec["url"], json=spec.get("body"), timeout=12)
        else:
            r = requests.get(x, timeout=12)
        return f"HTTP {r.status_code}\\n{r.text[:1400]}"
    except Exception as e:
        return f"Error: {e}"`); }
    if (tls.some((t) => t.id === "human_approval")) { reg.push(`"human_approval": tool_human_approval`); defs.push(`def tool_human_approval(x):
    ans = input(f"[APPROVAL NEEDED] {x} (yes/no) > ").strip().lower()
    return "User APPROVED." if ans in ("y", "yes", "approve") else "User DENIED the action."`); }
    if (tls.some((t) => t.id === "statistics")) { imports.push("import statistics as _st"); reg.push(`"statistics": tool_statistics`); defs.push(`def tool_statistics(x):
    nums = [float(n) for n in re.findall(r"-?\\d+(?:\\.\\d+)?", x)]
    if not nums:
        return "Error: provide numbers."
    return f"count={len(nums)} mean={_st.mean(nums):.3f} median={_st.median(nums)} min={min(nums)} max={max(nums)} stdev={_st.pstdev(nums):.3f}"`); }
    if (tls.some((t) => t.id === "unit_convert")) { reg.push(`"unit_convert": tool_unit_convert`); defs.push(`def tool_unit_convert(x):
    m = re.search(r"(-?\\d+(?:\\.\\d+)?)\\s*\\u00b0?\\s*([a-z]+)\\s*(?:to|in|->)\\s*\\u00b0?\\s*([a-z]+)", x.lower())
    if not m:
        return "Error: use like '10 km to mi'."
    v, a, b = float(m.group(1)), m.group(2), m.group(3)
    f = {"m":1,"km":1000,"cm":0.01,"mm":0.001,"mi":1609.344,"ft":0.3048,"in":0.0254,"yd":0.9144,"g":1,"kg":1000,"mg":0.001,"lb":453.592,"oz":28.3495}
    if a in f and b in f:
        return f"{v} {a} = {v*f[a]/f[b]:.4f} {b}"
    to_c = {"c": lambda t: t, "f": lambda t: (t-32)*5/9, "k": lambda t: t-273.15}
    fr_c = {"c": lambda t: t, "f": lambda t: t*9/5+32, "k": lambda t: t+273.15}
    if a in to_c and b in fr_c:
        return f"{v}{a.upper()} = {fr_c[b](to_c[a](v)):.2f}{b.upper()}"
    return "Error: unknown units."`); }
    if (tls.some((t) => t.id === "json_extract")) { if (!imports.includes("import json as _json")) imports.push("import json as _json"); reg.push(`"json_extract": tool_json_extract`); defs.push(`def tool_json_extract(x):
    parts = x.split("\\n", 1)
    path = parts[0].strip() if len(parts) == 2 else ""
    body = re.sub(r"^HTTP\\s+\\d+\\s*", "", parts[1] if len(parts) == 2 else x).strip()
    try:
        data = _json.loads(body)
    except Exception:
        return "Error: invalid JSON."
    if not path:
        return "keys: " + ", ".join(str(k) for k in (data.keys() if isinstance(data, dict) else range(len(data))))
    cur = data
    for k in [p for p in re.split(r"[.\\[\\]]", path) if p]:
        cur = cur[int(k)] if isinstance(cur, list) else cur.get(k)
        if cur is None:
            break
    return _json.dumps(cur)[:800] if isinstance(cur, (dict, list)) else str(cur)`); }
    return `# AI Workbench · ReAct agent "${name}"  (pip install openai${tls.some((t) => ["web_fetch", "http_request"].includes(t.id)) ? " requests" : ""})
import os
${imports.join("\n")}
from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = ${JSON.stringify(model || "your-model")}

# ── tools ──
${defs.join("\n\n")}

TOOLS = { ${reg.join(", ")} }

SYSTEM = """${q3(reactSystemPrompt(tls, goal))}"""

def run(task, max_iters=${maxIters}):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for _ in range(max_iters):
        text = client.chat.completions.create(model=MODEL, messages=messages, temperature=${temperature}).choices[0].message.content
        final = re.search(r"Final Answer:\\s*([\\s\\S]*)", text, re.I)
        if final:
            return final.group(1).strip()
        action = re.search(r"Action:\\s*([^\\n]+)", text, re.I)
        if not action:
            return text.strip()
        ainput = re.search(r"Action Input:\\s*([\\s\\S]*?)(?=\\n\\s*(?:Thought|Action|Observation)\\s*:|$)", text, re.I)
        name = action.group(1).strip()
        arg = ainput.group(1).strip() if ainput else ""
        obs = TOOLS.get(name, lambda a: f"Unknown tool: {name}")(arg)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    return "Stopped: reached the step limit."

if __name__ == "__main__":
    print(run(${JSON.stringify(task)}))`;
  }
  function copyCode() { navigator.clipboard.writeText(buildCode()).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }
  function downloadCode() { const blob = new Blob([buildCode()], { type: "text/x-python" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${(name || "agent").replace(/\s+/g, "_").toLowerCase()}.py`; a.click(); URL.revokeObjectURL(a.href); }
  function agentConfig() {
    return {
      type: agentType, name, description, systemPrompt: goal, provider: providerId, model, temperature, maxIterations: maxIters,
      tools: [...enabledTools], knowledge: enabledTools.has("knowledge") ? knowledgeText : undefined,
      steps: agentType === "workflow" ? steps.map((s) => ({ name: s.name, instruction: s.instruction })) : undefined, task,
    };
  }
  // Create-or-update the saved project, returning its id (or "" on failure).
  async function persist(): Promise<string> {
    if (savedProjectId) {
      const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: savedProjectId, name: name || "agent", config: agentConfig() }) });
      return r.ok ? savedProjectId : "";
    }
    const r = await fetch("/api/projects", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ lab: "agent", name: name || "agent", config: agentConfig() }) });
    const j = await r.json().catch(() => null);
    if (r.ok && j?.id) { setSavedProjectId(j.id); return j.id; }
    return "";
  }
  async function saveAgent() {
    setMsg("");
    try {
      const id = await persist();
      if (id) { setSaved(true); setTimeout(() => setSaved(false), 1600); } else setMsg("Could not save the agent.");
    } catch (e) { setMsg((e as Error).message); }
  }
  async function publishAgent() {
    setPublishing(true); setMsg("");
    try {
      const id = await persist();
      if (!id) { toast("Publish failed", "error"); return; }
      const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id, published: true }) });
      if (r.ok) setPublished(true);
      toast(r.ok ? `Published “${name}” — open it in the Workroom` : "Publish failed", r.ok ? "success" : "error");
    } catch { toast("Publish failed", "error"); }
    finally { setPublishing(false); }
  }
  async function unpublishAgent() {
    if (!savedProjectId) { setPublished(false); return; }
    setPublishing(true);
    try {
      const r = await fetch("/api/projects", { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ id: savedProjectId, published: false }) });
      if (r.ok) { setPublished(false); toast("Removed from the Workroom", "success"); } else toast("Could not unpublish", "error");
    } catch { toast("Could not unpublish", "error"); }
    finally { setPublishing(false); }
  }
  function exportJson() { const blob = new Blob([JSON.stringify(agentConfig(), null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${(name || "agent").replace(/\s+/g, "_").toLowerCase()}.json`; a.click(); URL.revokeObjectURL(a.href); }
  async function loadAgents() {
    if (loadOpen) { setLoadOpen(false); return; }
    try { const r = await fetch("/api/projects?lab=agent"); const j = await r.json(); setSavedAgents(j.projects || []); } catch { setSavedAgents([]); }
    setLoadOpen(true);
  }
  /* eslint-disable @typescript-eslint/no-explicit-any */
  function applyConfig(cfg: any, id?: string, pub?: boolean) {
    if (!cfg) return;
    setSavedProjectId(id || "");
    setPublished(!!pub);
    setAgentType(cfg.type === "workflow" ? "workflow" : "react");
    if (cfg.name) setName(String(cfg.name));
    if (cfg.description) setDescription(String(cfg.description));
    if (cfg.systemPrompt) setGoal(String(cfg.systemPrompt));
    if (cfg.provider) { setProviderId(String(cfg.provider)); fetch(`/api/models?providerId=${cfg.provider}`).then((r) => r.json()).then((j) => setModelList(j.models || [])).catch(() => {}); }
    if (cfg.model) setModel(String(cfg.model));
    if (typeof cfg.temperature === "number") setTemperature(cfg.temperature);
    if (cfg.maxIterations) setMaxIters(Math.max(1, Math.min(10, Number(cfg.maxIterations))));
    if (Array.isArray(cfg.tools)) { const s = new Set<string>(cfg.tools.filter((t: string) => AGENT_TOOLS.some((x) => x.id === t))); setEnabledTools(s); setPlacedTools(new Set(s)); }
    if (cfg.knowledge) setKnowledgeText(String(cfg.knowledge));
    if (Array.isArray(cfg.steps)) setSteps(cfg.steps.map((s: any, i: number) => ({ id: `s${i + 1}`, name: String(s.name || `Step ${i + 1}`), instruction: String(s.instruction || "") })));
    if (cfg.task) setTask(String(cfg.task));
    setBuildMode(cfg.type === "workflow" ? "manual" : "visual"); setNodePos({}); setNodeStatus({}); setASel("agent"); setLoadOpen(false); setStep("build");
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */

  // Load a saved agent when opened from My Projects (?project=<id>).
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("project");
    if (!id) return;
    fetch(`/api/projects?id=${id}`).then((r) => r.json()).then(({ project }) => { if (project?.config) applyConfig(project.config, project.id || id, project.published); }).catch(() => {});
  }, []);

  const curType = TYPES.find((t) => t.id === agentType)!;
  const stepBtn = (k: Step, n: number, label: string) => (<button className={step === k ? "on" : ""} onClick={() => setStep(k)}><b>{n}</b>{label}</button>);
  const selNode = nodes.find((n) => n.id === aSel);

  return (
    <>
      <div className="lab-head">
        <div><div className="eyebrow">Lab 03 · orchestration</div><h2 className="page-h">Agent Lab</h2><p className="page-sub" style={{ margin: 0 }}>Pick an agent type, wire it up on a node canvas (or by form / from a prompt), then run it and watch every step.</p></div>
        <div className="acts" style={{ position: "relative" }}>
          <div className="seg" style={{ width: 210, marginRight: 6 }}>
            <button className={runtime === "browser" ? "on" : ""} onClick={() => setRuntime("browser")}>In-browser</button>
            <button className={runtime === "nat" ? "on" : ""} onClick={() => setRuntime("nat")}>NVIDIA NAT</button>
          </div>
          {runtime === "browser" && <>
            <button className="btn ghost sm" onClick={loadAgents}>📂 Load</button>
            <button className="btn ghost sm" onClick={saveAgent}>{saved ? "Saved ✓" : "💾 Save"}</button>
            <button className="btn ghost sm" onClick={exportJson}>⬇ Export JSON</button>
            <button className="btn ghost sm" onClick={() => setShowCode(true)}>&lt;/&gt; Get code</button>
            {published
              ? <><Link className="btn ghost sm" href="/workroom" style={{ color: "#3b9e5f" }}>● Published</Link><button className="btn ghost sm" onClick={unpublishAgent} disabled={publishing} title="Remove from the Workroom">Unpublish</button></>
              : <button className="btn sm" onClick={publishAgent} disabled={publishing || !hasProvider} title="Make this agent usable in the Workroom">{publishing ? "Publishing…" : "🚀 Publish"}</button>}
          </>}
          {loadOpen && <div className="addmenu2" style={{ top: 38 }}><div className="hd">Saved agents</div>{savedAgents.length ? savedAgents.map((a) => <div key={a.id} className="ai" onClick={() => applyConfig(a.config, a.id, a.published)}>{a.name}{a.published ? " ●" : ""}</div>) : <div className="ai" style={{ color: "var(--faint)" }}>none saved yet</div>}</div>}
        </div>
      </div>
      {runtime === "nat" ? <NatAgentPanel /> : <>
      {provKnown && !hasProvider && <div className="warnbar">No provider configured — an admin must add one under Admin → Providers before you can run an agent.</div>}
      <div className="teach-note" style={{ marginBottom: 12 }}><span className="ic">🔌</span><span>To attach <b>MCP servers</b> or <b>knowledge bases</b> to an agent, switch to the <b>NVIDIA NAT</b> runtime above — the in-browser runtime uses built-in tools only.</span></div>
      {msg && <div className="err">{msg}</div>}
      <input ref={fileRef} type="file" accept=".txt,.md,.csv,.pdf,.docx,.doc,.xlsx,.xls" onChange={onKnowledgeFile} style={{ display: "none" }} />

      <div className="stepper">{stepBtn("type", 1, "Type")}{stepBtn("build", 2, "Build")}{stepBtn("run", 3, "Run")}{stepBtn("learn", 4, "Learn")}</div>

      {/* STEP 1 — TYPE */}
      {step === "type" && (<>
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="card-h"><span className="t">When do you actually need an agent?</span></div>
          <div className="card-b">
            <div className="whenuse">
              <div className="wu step-1">
                <div className="wu-head"><span className="wu-step">Step 1 · Direct</span></div>
                <b>Single prompt</b>
                <span>One question, one answer, no tools or steps. Cheapest &amp; fastest — use the Prompting Lab.</span>
              </div>
              <div className="wu step-2">
                <div className="wu-head"><span className="wu-step">Step 2 · Pipeline</span></div>
                <b>Workflow</b>
                <span>A fixed sequence you always run in the same order (plan → draft → review). Predictable &amp; debuggable.</span>
              </div>
              <div className="wu step-3">
                <div className="wu-head"><span className="wu-step">Step 3 · Autonomous</span></div>
                <b>Tool agent (ReAct)</b>
                <span>The task needs live facts, math, APIs, or decisions the model must make itself. Powerful, but slower &amp; pricier — you trade control for autonomy.</span>
              </div>
            </div>
          </div>
        </div>
        <div className="card">
          <div className="card-h"><span className="t">Choose an agent type</span></div>
          <div className="card-b">
            <div className="model-grid">
              {TYPES.map((t) => (
                <div key={t.id} className={`model-card ${agentType === t.id ? "on" : ""}`} onClick={() => { setAgentType(t.id); setBuildMode(t.id === "react" ? "visual" : "manual"); }}>
                  <div className="mc-top"><span className="mc-name">{t.label}</span><span className="mc-fam">{t.tag}</span></div>
                  <div className="mc-desc">{t.desc}</div>
                  <div className="mc-foot">{agentType === t.id ? "selected" : "click to select"}</div>
                </div>
              ))}
            </div>
            <div className="stepnav"><button className="btn" onClick={() => setStep("build")}>Build from scratch →</button></div>
          </div>
        </div>
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-h"><span className="t">…or start from a template</span><span className="mono r">real, editable</span></div>
          <div className="card-b">
            <div className="model-grid">
              {TEMPLATES.map((t) => (
                <div key={t.id} className="model-card" onClick={() => applyTemplate(t)}>
                  <div className="mc-top"><span className="mc-name">{t.label}</span><span className="mc-fam">{t.tag}</span></div>
                  <div className="mc-desc">{t.desc}</div>
                  <div className="mc-foot">{t.tools.map((x) => TOOL_META[x]?.label).join(" · ")}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </>)}

      {/* STEP 2 — BUILD */}
      {step === "build" && (<>
        {agentType === "react" && <div className="seg" style={{ maxWidth: 420, marginBottom: 16 }}>
          <button className={buildMode === "visual" ? "on" : ""} onClick={() => setBuildMode("visual")}>Visual builder</button>
          <button className={buildMode === "manual" ? "on" : ""} onClick={() => setBuildMode("manual")}>Manual</button>
          <button className={buildMode === "prompt" ? "on" : ""} onClick={() => setBuildMode("prompt")}>From prompt</button>
        </div>}

        {/* REACT · VISUAL NODE CANVAS */}
        {agentType === "react" && buildMode === "visual" && (
          <div className={`agent-flow ${fullscreen ? "fs" : ""}`}>
          <div className="split" style={{ gridTemplateColumns: "1fr 320px" }}>
            <div className="card">
              <div className="card-h"><span className="t">Flow</span>
                <div className="r" style={{ position: "relative" }}>
                  <button className="btn ghost sm" onClick={() => setFullscreen((f) => !f)}>{fullscreen ? "⤢ Exit" : "⛶ Fullscreen"}</button>
                  <button className="btn ghost sm" onClick={() => setAddOpen((o) => !o)}>+ Add node</button>
                  <button className="btn sm" onClick={startRun} disabled={!hasProvider}>▶ Run</button>
                  {addOpen && (
                    <div className="addmenu2">
                      <div className="hd">Add a tool node to the canvas</div>
                      {AGENT_TOOLS.map((t) => { const on = placedTools.has(t.id); return <div key={t.id} className="ai" onClick={() => { togglePlaced(t.id); if (!on) setASel("tool:" + t.id); }}><span>{TOOL_META[t.id]?.icon ?? "🔧"}</span>{TOOL_META[t.id]?.label ?? t.id}<span className={`ai-state ${on ? "on" : ""}`}>{on ? "✓ on canvas" : "+ add"}</span></div>; })}
                    </div>
                  )}
                </div>
              </div>
              <div className="card-b" style={{ padding: 0 }}>{flowCanvas()}</div>
            </div>
            <div className="card">
              <div className="card-h"><span className="t">Configure</span><span className="mono r">{selNode?.title || "—"}</span></div>
              <div className="card-b" style={{ maxHeight: CANVAS_H, overflow: "auto" }}>
                {!selNode && <div className="note">Click a node to configure it.</div>}
                {selNode?.type === "trigger" && <div className="note">This is where the user&apos;s task enters the agent. You&apos;ll type the actual task on the Run step.</div>}
                {selNode?.type === "output" && <div className="note">The agent&apos;s Final Answer is returned here after the reasoning loop finishes.</div>}
                {selNode?.type === "agent" && (<>
                  <div className="insp-field"><div className="k">Agent name</div><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Description</div><input type="text" value={description} onChange={(e) => setDescription(e.target.value)} /></div>
                  <div className="insp-field"><div className="k">System prompt (role & goal)</div><textarea rows={4} value={goal} onChange={(e) => setGoal(e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Max reasoning steps · {maxIters}</div><input type="range" min={1} max={10} value={maxIters} onChange={(e) => setMaxIters(+e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Connected</div><div className="chip-row"><span className="c">{model || providerLabel}</span>{[...enabledTools].map((t) => <span key={t} className="c">{TOOL_META[t]?.label}</span>)}</div></div>
                </>)}
                {selNode?.type === "model" && (<>
                  <div className="insp-field"><div className="k">Provider</div>
                    <select value={providerId} onChange={(e) => loadModels(e.target.value)}>{providers.length ? providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>) : <option>none configured</option>}</select>
                  </div>
                  <div className="insp-field"><div className="k">Model</div>
                    {modelList.length ? <select value={model} onChange={(e) => setModel(e.target.value)}>{modelList.map((m) => <option key={m}>{m}</option>)}</select> : <input type="text" value={model} onChange={(e) => setModel(e.target.value)} placeholder="model id" />}
                  </div>
                  <div className="insp-field"><div className="k">Temperature · {temperature}</div><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div>
                  <div className="insp-field"><div className="k">Max tokens</div><input type="text" value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value) || 600)} /></div>
                </>)}
                {selNode?.type === "tool" && selNode.toolId !== "knowledge" && (<>
                  <div className="insp-field"><div className="k">Tool</div><div className="note" style={{ margin: 0 }}>{AGENT_TOOLS.find((t) => t.id === selNode.toolId)?.desc}</div></div>
                  <div className="insp-field"><div className="k">Example input the agent sends</div><div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>{AGENT_TOOLS.find((t) => t.id === selNode.toolId)?.example}</div></div>
                  <div className="row" style={{ gap: 8 }}>
                    {enabledTools.has(selNode.toolId!) ? <button className="btn ghost sm" onClick={() => disconnectTool(selNode.toolId!)}>Disconnect</button> : <button className="btn sm" onClick={() => connectTool(selNode.toolId!)}>Connect to agent</button>}
                    <button className="btn ghost sm" onClick={() => { removeToolNode(selNode.toolId!); setASel("agent"); }}>Remove node</button>
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>Drag its top dot to the Agent to wire it, or click a wire and press <b>Delete</b>. {enabledTools.has(selNode.toolId!) ? "" : "Unconnected tools aren't used at run."}</div>
                </>)}
                {selNode?.type === "tool" && selNode.toolId === "knowledge" && (<>
                  <div className="insp-field"><div className="k">Knowledge base (the agent searches this)</div><textarea rows={5} value={knowledgeText} onChange={(e) => setKnowledgeText(e.target.value)} /></div>
                  <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                    <button className="btn ghost sm" onClick={() => fileRef.current?.click()} disabled={uploading}>{uploading ? "Parsing…" : "Upload doc"}</button>
                    {enabledTools.has("knowledge") ? <button className="btn ghost sm" onClick={() => disconnectTool("knowledge")}>Disconnect</button> : <button className="btn sm" onClick={() => connectTool("knowledge")}>Connect</button>}
                    <button className="btn ghost sm" onClick={() => { removeToolNode("knowledge"); setASel("agent"); }}>Remove</button>
                  </div>
                  <div className="note" style={{ marginTop: 8 }}>{knowledgeText.split(/\s+/).filter(Boolean).length} words · pdf / docx / xlsx supported</div>
                </>)}
              </div>
            </div>
          </div>
          </div>
        )}

        {/* REACT · MANUAL FORM */}
        {agentType === "react" && buildMode === "manual" && (
          <div className="card"><div className="card-h"><span className="t">Agent configuration</span></div>
            <div className="card-b">
              <div className="split col-2e"><div><label className="fld">Agent name</label><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></div><div><label className="fld">Description</label><input type="text" value={description} onChange={(e) => setDescription(e.target.value)} /></div></div>
              <label className="fld" style={{ marginTop: 12 }}>System prompt (role & goal)</label><textarea rows={3} value={goal} onChange={(e) => setGoal(e.target.value)} />
              <div className="split col-2e" style={{ marginTop: 12 }}>
                <div><label className="fld">Provider</label><select value={providerId} onChange={(e) => loadModels(e.target.value)}>{providers.length ? providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>) : <option>none</option>}</select></div>
                <div><label className="fld">Model</label>{modelList.length ? <select value={model} onChange={(e) => setModel(e.target.value)}>{modelList.map((m) => <option key={m}>{m}</option>)}</select> : <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />}</div>
              </div>
              <div className="split col-2e" style={{ marginTop: 12 }}><div><label className="fld">Temperature · {temperature}</label><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div><div><label className="fld">Max reasoning steps · {maxIters}</label><input type="range" min={1} max={10} value={maxIters} onChange={(e) => setMaxIters(+e.target.value)} /></div></div>
              <label className="fld" style={{ marginTop: 12 }}>Tools</label>
              <div className="checklist">{AGENT_TOOLS.map((t) => <span key={t.id} className={`chk ${enabledTools.has(t.id) ? "on" : ""}`} onClick={() => toggleManual(t.id)} title={t.desc}>{t.name}</span>)}</div>
              {enabledTools.has("knowledge") && (<>
                <label className="fld" style={{ marginTop: 12 }}>Knowledge base</label>
                <textarea rows={3} value={knowledgeText} onChange={(e) => setKnowledgeText(e.target.value)} />
                <div className="row" style={{ marginTop: 8, gap: 8 }}><button className="btn ghost sm" onClick={() => fileRef.current?.click()} disabled={uploading}>{uploading ? "Parsing…" : "Upload doc (pdf/docx/xlsx)"}</button><span className="note">{knowledgeText.split(/\s+/).filter(Boolean).length} words loaded</span></div>
              </>)}
              <div className="stepnav"><button className="btn ghost" onClick={() => setStep("type")}>← Back</button><button className="btn" onClick={startRun} disabled={!hasProvider}>Next: Run →</button></div>
            </div>
          </div>
        )}

        {/* FROM PROMPT (all types) */}
        {(buildMode === "prompt" || agentType !== "react") && (
          <>
            {agentType === "react" && buildMode === "prompt" && (
              <div className="card" style={{ marginBottom: 16 }}><div className="card-h"><span className="t">Describe your agent</span></div>
                <div className="card-b"><label className="fld">Tell the builder what the agent should do — the LLM drafts the config</label><textarea rows={3} value={describe} onChange={(e) => setDescribe(e.target.value)} /><div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={generateFromPrompt} disabled={generating || !hasProvider}>{generating ? <><span className="busy-dot" />Generating…</> : "✦ Generate config"}</button><span className="note">fills the Visual builder</span></div></div>
              </div>
            )}
            {agentType === "workflow" && (<>
              <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-h"><span className="t">Pipeline</span><span className="mono r">{steps.length} steps</span></div>
                <div className="card-b" style={{ padding: 0 }}>{wfCanvas()}</div>
              </div>
              <div className="card"><div className="card-h"><span className="t">Workflow steps</span></div>
                <div className="card-b">
                  <div className="split col-2e"><div><label className="fld">Name</label><input type="text" value={name} onChange={(e) => setName(e.target.value)} /></div><div><label className="fld">Temperature · {temperature}</label><input type="range" min={0} max={1} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} /></div></div>
                  <div className="split col-2e" style={{ marginTop: 12 }}>
                    <div><label className="fld">Provider</label><select value={providerId} onChange={(e) => loadModels(e.target.value)}>{providers.length ? providers.map((p) => <option key={p.id} value={p.id}>{p.label || p.provider}</option>) : <option>none</option>}</select></div>
                    <div><label className="fld">Model</label>{modelList.length ? <select value={model} onChange={(e) => setModel(e.target.value)}>{modelList.map((m) => <option key={m}>{m}</option>)}</select> : <input type="text" value={model} onChange={(e) => setModel(e.target.value)} />}</div>
                  </div>
                  <label className="fld" style={{ marginTop: 12 }}>Steps (run top to bottom, each sees the previous output)</label>
                  {steps.map((s, i) => (
                    <div key={s.id} className="wf-edit"><div className="wf-idx">{i + 1}</div><input type="text" value={s.name} onChange={(e) => setSteps((ss) => ss.map((x) => x.id === s.id ? { ...x, name: e.target.value } : x))} style={{ maxWidth: 150 }} /><input type="text" value={s.instruction} onChange={(e) => setSteps((ss) => ss.map((x) => x.id === s.id ? { ...x, instruction: e.target.value } : x))} /><button className="x" onClick={() => setSteps((ss) => ss.filter((x) => x.id !== s.id))}>×</button></div>
                  ))}
                  <button className="btn ghost sm" style={{ marginTop: 8 }} onClick={() => setSteps((ss) => [...ss, { id: `s${Date.now()}`, name: `Step ${ss.length + 1}`, instruction: "" }])}>+ Add step</button>
                  <div className="stepnav"><button className="btn ghost" onClick={() => setStep("type")}>← Back</button><button className="btn" onClick={startRun} disabled={!hasProvider}>Next: Run →</button></div>
                </div>
              </div>
            </>)}
          </>
        )}

        {agentType === "react" && buildMode === "visual" && <div className="stepnav"><button className="btn ghost" onClick={() => setStep("type")}>← Back</button><button className="btn" onClick={startRun} disabled={!hasProvider}>Next: Run →</button></div>}
      </>)}

      {/* STEP 3 — RUN */}
      {step === "run" && (<>
        {agentType === "react" && (<>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h"><span className="t">Execution flow</span><span className="mono r">{running ? "running…" : finalOut ? "finished ✓" : "idle"}</span></div>
            <div className="card-b" style={{ padding: 0 }}>{flowCanvas()}</div>
          </div>
          {pendingApproval && (
            <div className="approval">
              <div className="ap-q"><span className="ap-ic">🙋</span><div><div className="ap-title">Human approval needed</div><div className="ap-text">{pendingApproval.q}</div></div></div>
              <div className="row" style={{ gap: 8 }}><button className="btn" onClick={() => { pendingApproval.resolve("User APPROVED."); setPendingApproval(null); }}>✓ Approve</button><button className="btn ghost" onClick={() => { pendingApproval.resolve("User DENIED the action."); setPendingApproval(null); }}>✗ Deny</button></div>
            </div>
          )}
          {metrics && !running && <div className="ag-metrics"><span className="m"><b>{metrics.calls}</b> LLM calls</span><span className="m"><b>{metrics.tools}</b> tool calls</span><span className="m"><b>{(metrics.ms / 1000).toFixed(1)}s</b> latency</span><span className="m"><b>~{metrics.tokens >= 1000 ? (metrics.tokens / 1000).toFixed(1) + "k" : metrics.tokens}</b> tokens</span></div>}
          <div className="split col-2">
            <div className="card"><div className="card-h"><span className="t">Task</span></div>
              <div className="card-b">
                <label className="fld">What should {name} do?</label><textarea rows={3} value={task} onChange={(e) => setTask(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!running && hasProvider && task.trim()) runReact(); } }} placeholder="Type your prompt and press Enter to run…" />
                <div className="row" style={{ marginTop: 12 }}><button className="btn" onClick={runReact} disabled={running || !hasProvider}>{running ? <><span className="busy-dot" />Running…</> : "▶ Run agent"}</button><span className="note">{toolList.length} tools · {model || providerLabel} · max {maxIters} steps</span></div>
                {finalOut && <><label className="fld" style={{ marginTop: 16 }}>Final answer</label><div className="out" style={{ position: "relative" }}><CopyBtn text={formatFinalAnswer(finalOut)} /><div style={{ paddingRight: 36, whiteSpace: "pre-wrap", fontFamily: "var(--mono)", fontSize: "12.5px" }}>{formatFinalAnswer(finalOut)}</div></div></>}
              </div>
            </div>
            <div className="card"><div className="card-h"><span className="t">Reasoning trace</span><span className="mono r">{trace.length} steps</span></div>
              <div className="card-b" ref={scrollRef} style={{ maxHeight: 460, overflow: "auto" }}>
                {trace.length === 0 && <div className="note">Run the agent to watch the Thought → Action → Observation loop.</div>}
                {trace.map((t, i) => (<div key={i} className={`ag-step ${t.kind} ${t.state}`}><div className="ag-k">{t.kind === "action" ? `action · ${t.tool}` : t.kind === "final" ? "final answer" : t.kind}</div><div className="ag-t">{t.kind === "thought" && t.state === "active" ? <><span className="busy-dot" />{t.text}</> : t.text}</div></div>))}
              </div>
            </div>
          </div>
        </>)}
        {agentType === "workflow" && (<>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-h"><span className="t">Execution flow</span><span className="mono r">{running ? "running…" : finalOut ? "finished ✓" : "idle"}</span></div>
            <div className="card-b" style={{ padding: 0 }}>{wfCanvas()}</div>
          </div>
          <div className="card"><div className="card-h"><span className="t">Workflow · {name}</span><span className="mono r">{steps.length} steps</span></div>
            <div className="card-b" ref={scrollRef} style={{ maxHeight: 560, overflow: "auto" }}>
              <label className="fld">Input</label><textarea rows={2} value={task} onChange={(e) => setTask(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!running && hasProvider && task.trim()) runWorkflow(); } }} placeholder="Type your prompt and press Enter to run…" />
              <div className="row" style={{ margin: "12px 0" }}><button className="btn" onClick={runWorkflow} disabled={running || !hasProvider}>{running ? <><span className="busy-dot" />Running…</> : "▶ Run workflow"}</button></div>
              {wfOutputs.map((o, i) => (<div key={i} className={`wf-run ${o.state}`}><div className="wf-run-h"><span className="wf-idx">{i + 1}</span><b>{o.name}</b>{o.state === "active" && <span className="busy-dot" style={{ marginLeft: 8 }} />}{o.state === "done" && <span className="badge good" style={{ marginLeft: 8 }}>done</span>}</div>{o.text && <div className="out" style={{ marginTop: 8, position: "relative" }}><CopyBtn text={formatFinalAnswer(o.text)} /><div style={{ paddingRight: 36, whiteSpace: "pre-wrap", fontFamily: "var(--mono)", fontSize: "12.5px" }}>{formatFinalAnswer(o.text)}</div></div>}</div>))}
            </div>
          </div>
        </>)}
        <div className="stepnav"><button className="btn ghost" onClick={() => setStep("build")}>← Back to build</button></div>
      </>)}

      {/* STEP 4 — LEARN NETWORK GRAPH & AI AGENT TUTOR */}
      {step === "learn" && (<>
        {!selectedTopic ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            {/* FORMAL AI AGENT CONCEPT NETWORK TREE MAP ARCHITECTURE */}
            <div className="card" style={{ borderRadius: 16 }}>
              {/* Header Banner */}
              <div className="card-h">
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 15, color: "var(--text)" }}>
                  <MapSvg size={16} color="var(--accent-strong)" /> <b>AI Agent Concept Network Map</b>
                </span>
                <span className="badge good">{LEARN_MAP.length} Interactive Lessons</span>
              </div>

              <div className="card-b" style={{
                padding: "24px 20px",
                background: "radial-gradient(var(--border) 1px, transparent 1px)",
                backgroundSize: "18px 18px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 18
              }}>
                {/* 1. Root Master AI Agent Node */}
                <div style={{
                  background: "var(--surface)",
                  border: "1.5px solid var(--accent)",
                  borderRadius: 24,
                  padding: "9px 24px",
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  boxShadow: "var(--shadow-md)",
                  zIndex: 2
                }}>
                  <BotSvg size={18} color="var(--accent-strong)" />
                  <span style={{ fontWeight: 800, fontSize: 14.5, color: "var(--text)", letterSpacing: "-0.01em" }}>AI Agent Architecture</span>
                </div>

                {/* 2. Top Branch Connecting Lines (SVG Tree Lines) */}
                <div style={{ width: "100%", height: 26, display: "flex", justifyContent: "center" }}>
                  <svg width="100%" height="26" style={{ overflow: "visible" }}>
                    <path
                      d="M 50% 0 L 50% 13 M 10% 13 L 90% 13 M 10% 13 L 10% 26 M 30% 13 L 30% 26 M 50% 13 L 50% 26 M 70% 13 L 70% 26 M 90% 13 L 90% 26"
                      fill="none"
                      stroke="var(--border-strong)"
                      strokeWidth="1.5"
                      strokeDasharray="4 3"
                    />
                  </svg>
                </div>

                {/* 3. 5 Category Tree Columns */}
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
                  gap: 14,
                  width: "100%"
                }}>
                  {CURRICULUM_CATEGORIES.map((cat) => (
                    <div
                      key={cat.id}
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 10
                      }}
                    >
                      {/* Category Tree Node */}
                      <div style={{
                        width: "100%",
                        background: cat.bg,
                        border: `1.5px solid ${cat.border}`,
                        borderRadius: 10,
                        padding: "9px 11px",
                        display: "flex",
                        flexDirection: "column",
                        gap: 3,
                        textAlign: "center",
                        alignItems: "center",
                        boxShadow: `0 2px 8px ${cat.color}15`
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          {cat.getSvgIcon(cat.color)}
                          <span style={{ fontWeight: 800, fontSize: 12.5, color: "var(--text)" }}>{cat.label}</span>
                        </div>
                        <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 500 }}>{cat.tagline}</span>
                      </div>

                      {/* Sub-tree Down Connector Line */}
                      <div style={{ width: 1.5, height: 12, background: cat.color, opacity: 0.6 }} />

                      {/* Vertical Stack of Topic Node Cards */}
                      <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
                        {cat.getTopics().map((topic, idx, arr) => (
                          <React.Fragment key={topic.id}>
                            <button
                              style={{
                                background: "var(--surface)",
                                border: `1px solid ${cat.border}`,
                                borderRadius: 9,
                                padding: "8px 10px",
                                display: "flex",
                                alignItems: "center",
                                gap: 9,
                                textAlign: "left",
                                cursor: "pointer",
                                transition: "all 0.15s ease",
                                width: "100%"
                              }}
                              onClick={() => setSelectedTopic(topic)}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.borderColor = cat.color;
                                e.currentTarget.style.transform = "translateY(-1px)";
                                e.currentTarget.style.boxShadow = `0 4px 12px ${cat.color}22`;
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.borderColor = cat.border;
                                e.currentTarget.style.transform = "none";
                                e.currentTarget.style.boxShadow = "none";
                              }}
                            >
                              <div style={{
                                width: 26,
                                height: 26,
                                borderRadius: 6,
                                background: cat.bg,
                                border: `1px solid ${cat.border}`,
                                display: "grid",
                                placeItems: "center",
                                flexShrink: 0,
                                color: cat.color
                              }}>
                                {topic.id === "types_of_agents" ? <PuzzleSvg size={13} color={cat.color} /> :
                                 topic.id === "genai_vs_agent" ? <ScaleSvg size={13} color={cat.color} /> :
                                 topic.id === "llm_plus_apis" ? <ZapSvg size={13} color={cat.color} /> :
                                 topic.id === "message_flow" || topic.id === "react_loop" ? <RefreshSvg size={13} color={cat.color} /> :
                                 topic.id === "chunks_and_embeddings" ? <BookSvg size={13} color={cat.color} /> :
                                 topic.id === "5_step_build" ? <BuildSvg size={13} color={cat.color} /> :
                                 <BotSvg size={13} color={cat.color} />}
                              </div>
                              <div style={{ display: "flex", flexDirection: "column", gap: 2, flex: 1, minWidth: 0, overflow: "hidden" }}>
                                <div style={{ fontWeight: 700, fontSize: 11.5, color: "var(--text)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", lineHeight: 1.25 }}>
                                  {topic.title}
                                </div>
                                <div style={{ fontSize: 10, color: cat.color, fontWeight: 700, display: "flex", alignItems: "center", gap: 3 }}>
                                  Click to learn →
                                </div>
                              </div>
                            </button>
                            {idx < arr.length - 1 && (
                              <div style={{ width: 1.5, height: 8, background: cat.color, margin: "0 auto", opacity: 0.4 }} />
                            )}
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Dedicated AI Agent Tutor Chatbot */}
            <div className="card" style={{ borderRadius: 16, border: "1.5px solid rgba(99,102,241,0.35)", background: "var(--surface)" }}>
                <div className="card-h">
                  <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                    <BotSvg size={16} color="var(--accent-strong)" /> <b>AI Agent Tutor — Clarify Any Doubt</b>
                  </span>
                  <span className="badge good">Grounded Knowledge Tutor</span>
                </div>
                <div className="card-b">
                  <div style={{ fontSize: 12.5, color: "var(--muted)", marginBottom: 12 }}>
                    Ask me any question about <b>What is an AI Agent, Generative AI vs AI Agent, Chunks, Embeddings, RAG, APIs, LLMs, or Implementation steps</b>. 
                    <i>(Note: This tutor is strictly guardrailed to answer AI Agent topics only.)</i>
                  </div>

                  {/* Preset Quick Questions */}
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
                    {[
                      "Difference between Generative AI and AI Agent?",
                      "How do Chunks & Embeddings work in RAG?",
                      "Why do we use LLMs and APIs together?",
                      "How does an agent understand messages?",
                      "What are the 5 steps to build an agent?",
                    ].map((q, idx) => (
                      <button
                        key={idx}
                        className="btn ghost sm"
                        style={{ fontSize: 11.5, borderRadius: 20, padding: "4px 10px", borderColor: "var(--accent-weak)" }}
                        onClick={() => askTutor(q)}
                      >
                        💡 {q}
                      </button>
                    ))}
                  </div>

                  {/* Messages Area */}
                  <div style={{
                    maxHeight: 280, overflowY: "auto", background: "var(--panel)", borderRadius: 12, padding: 14,
                    display: "flex", flexDirection: "column", gap: 10, border: "1px solid var(--border)", marginBottom: 14
                  }}>
                    {tutorMessages.map((m, i) => (
                      <div
                        key={i}
                        style={{
                          alignSelf: m.sender === "user" ? "flex-end" : "flex-start",
                          maxWidth: m.sender === "user" ? "80%" : "92%",
                          background: m.sender === "user" ? "var(--accent)" : "var(--surface)",
                          color: m.sender === "user" ? "#fff" : "var(--text)",
                          padding: "10px 14px",
                          borderRadius: 12,
                          border: m.sender === "bot" ? "1px solid var(--border)" : "none",
                          fontSize: 12.5,
                          lineHeight: 1.6,
                          boxShadow: "0 2px 8px rgba(0,0,0,0.06)"
                        }}
                      >
                        {renderTutorMessage(m.text)}
                      </div>
                    ))}
                  </div>

                  {/* Input Row */}
                  <div style={{ display: "flex", gap: 8 }}>
                    <input
                      type="text"
                      className="fld"
                      style={{ flex: 1, borderRadius: 10, padding: "8px 12px", fontSize: 13 }}
                      placeholder="Type your question about AI Agents (e.g. How does a ReAct loop work?)..."
                      value={tutorInput}
                      onChange={(e) => setTutorInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && tutorInput.trim()) askTutor(tutorInput); }}
                    />
                    <button
                      className="btn"
                      style={{ padding: "8px 16px", borderRadius: 10, fontWeight: 700 }}
                      onClick={() => { if (tutorInput.trim()) askTutor(tutorInput); }}
                    >
                      Ask Tutor →
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
          <div className="learn-detail" style={{ animation: "stepSlideIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) both" }}>
            {/* Navigation Bar */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <button className="btn ghost sm" onClick={() => { setSelectedTopic(null); setQuizOption(null); }} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span>←</span> <b>Back to Concept Network Map</b>
              </button>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="badge" style={{ background: "rgba(16,185,129,0.15)", color: "#10b981", border: "1px solid rgba(16,185,129,0.3)", padding: "4px 12px", borderRadius: 20, fontWeight: 700 }}>
                  🎓 Student Learning & Mastery Academy
                </span>
              </div>
            </div>

            {/* Hero Header */}
            <div className="card" style={{
              marginBottom: 20,
              background: "linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.12) 50%, rgba(16,185,129,0.1) 100%)",
              border: "1px solid rgba(99,102,241,0.35)",
              boxShadow: "0 12px 36px rgba(99,102,241,0.18)",
              borderRadius: 16,
              padding: 24
            }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 18 }}>
                <div style={{
                  fontSize: 40, width: 72, height: 72, borderRadius: 18, display: "grid", placeItems: "center",
                  background: "var(--surface)", border: "2px solid var(--accent)", boxShadow: "0 8px 24px rgba(99,102,241,0.25)", flexShrink: 0
                }}>
                  {selectedTopic.icon}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <h2 style={{ margin: 0, fontSize: 24, fontWeight: 800, color: "var(--text)" }}>{selectedTopic.title}</h2>
                      <span className="badge good" style={{ fontSize: 11, padding: "4px 8px" }}>{selectedTopic.category}</span>
                    </div>

                    <button
                      className="btn"
                      style={{ background: "linear-gradient(135deg, var(--accent), #7c3aed)", padding: "8px 16px", borderRadius: 10, fontWeight: 800, fontSize: 13 }}
                      onClick={() => { setTutorialActive(true); setTutorialStepIdx(0); }}
                    >
                      ▶ Start Interactive AI Guided Tutorial
                    </button>
                  </div>
                  <p style={{ margin: "8px 0 12px", fontSize: 15.5, fontWeight: 800, color: "var(--accent-strong)" }}>{selectedTopic.tagline}</p>
                  <p style={{ margin: 0, fontSize: 14.5, color: "var(--text)", lineHeight: 1.75, fontWeight: 500 }}>
                    <b style={{ color: "var(--accent-strong)", fontWeight: 800 }}>Core Meaning:</b> {selectedTopic.meaning}
                  </p>
                </div>
              </div>
            </div>

            {/* Interactive Guided AI Tutorial Modal/Player */}
            {tutorialActive && selectedTopic && (() => {
              const tutorialSteps = [
                {
                  title: "Lesson 1: Real-World Intuition & Analogy",
                  icon: "💡",
                  content: `To easily understand ${selectedTopic.title}, think of this real-world analogy:

${selectedTopic.meaning}

Key Takeaway: Rather than treating AI as a static text box, we treat it as a goal-oriented system capable of continuous reasoning!`
                },
                {
                  title: "Lesson 2: Step-by-Step Mechanism Breakdown",
                  icon: "⚙️",
                  content: `Here is exactly what happens step-by-step under the hood:

${selectedTopic.flowSteps.map((s, i) => `${i + 1}. ${s.label}: ${s.detail}`).join("\n\n")}`
                },
                {
                  title: "Lesson 3: Industry Case Study & Impact",
                  icon: "🚀",
                  content: `How top AI companies apply ${selectedTopic.title} in production:

${selectedTopic.realExamples.map((ex) => `• ${ex.name}: ${ex.desc}`).join("\n\n")}

Business Impact: ${selectedTopic.applications[0]?.impact || "Accelerates automation velocity."}`
                },
                {
                  title: "Lesson 4: Senior Developer Best Practices",
                  icon: "🎯",
                  content: `Golden rules for developers implementing ${selectedTopic.title}:

${selectedTopic.pitfallsAndMistakes.map((pm) => `❌ Mistake: ${pm.mistake}\n✅ Production Fix: ${pm.fix}`).join("\n\n")}

Pro Tip: ${selectedTopic.proTips[0] || "Always test agent behavior against edge cases before production deploy."}`
                }
              ];

              const curStep = tutorialSteps[tutorialStepIdx];
              const progress = Math.round(((tutorialStepIdx + 1) / tutorialSteps.length) * 100);

              return (
                <div className="card" style={{
                  marginBottom: 20, borderRadius: 16,
                  background: "linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.15))",
                  border: "2px solid var(--accent)",
                  boxShadow: "0 12px 36px rgba(99,102,241,0.25)"
                }}>
                  <div className="card-h" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                      <span>🎓</span> <b>AI Guided Tutorial — Lesson {tutorialStepIdx + 1} of {tutorialSteps.length}</b>
                    </span>
                    <button className="btn ghost sm" onClick={() => setTutorialActive(false)}>✕ Close Tutorial</button>
                  </div>
                  <div className="card-b">
                    {/* Progress Bar */}
                    <div style={{ height: 6, width: "100%", background: "var(--panel)", borderRadius: 4, overflow: "hidden", marginBottom: 16 }}>
                      <div style={{ height: "100%", width: `${progress}%`, background: "linear-gradient(90deg, var(--accent), #10b981)", transition: "width 0.3s ease" }} />
                    </div>

                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                      <span style={{ fontSize: 24 }}>{curStep.icon}</span>
                      <h3 style={{ margin: 0, fontSize: 16.5, fontWeight: 800, color: "var(--text)" }}>{curStep.title}</h3>
                    </div>

                    <div style={{
                      fontSize: 13.5, color: "var(--text)", lineHeight: 1.7, whiteSpace: "pre-wrap",
                      background: "var(--surface)", padding: 16, borderRadius: 12, border: "1px solid var(--border)", marginBottom: 16
                    }}>
                      {curStep.content}
                    </div>

                    {/* Step Navigation Buttons */}
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <button
                        className="btn ghost sm"
                        disabled={tutorialStepIdx === 0}
                        onClick={() => setTutorialStepIdx((i) => Math.max(0, i - 1))}
                      >
                        ← Previous Lesson
                      </button>
                      <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 600 }}>{progress}% Completed</span>
                      {tutorialStepIdx < tutorialSteps.length - 1 ? (
                        <button
                          className="btn sm"
                          onClick={() => setTutorialStepIdx((i) => i + 1)}
                        >
                          Next Lesson →
                        </button>
                      ) : (
                        <button
                          className="btn sm"
                          style={{ background: "#10b981" }}
                          onClick={() => { setTutorialActive(false); setTutorialStepIdx(0); }}
                        >
                          Finish Tutorial 🎉
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* TYPES OF AI AGENTS — FORMAL CLASSIFICATION MAP */}
            {selectedTopic.id === "types_of_agents" && (
              <div className="card" style={{ marginBottom: 20, borderRadius: 16 }}>
                <div className="card-h">
                  <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                    <PuzzleSvg size={16} color="var(--accent-strong)" /> <b>Types of AI Agents — Architecture Classification</b>
                  </span>
                  <span className="badge good">7 Core Agent Architectures</span>
                </div>
                <div className="card-b" style={{ padding: 20 }}>
                  <div style={{
                    width: "100%",
                    background: "var(--surface)",
                    borderRadius: 14,
                    padding: "20px 18px",
                    border: "1px solid var(--border)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 16
                  }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid var(--border)", paddingBottom: 12 }}>
                      <div>
                        <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent-strong)", fontWeight: 700 }}>Architectural Spectrum</div>
                        <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text)", marginTop: 2 }}>
                          7 Core AI Agent Classifications
                        </div>
                      </div>
                      <span className="mono r" style={{ opacity: 0.8 }}>Rule-based to Swarm</span>
                    </div>

                    {/* 7 Formal Agent Type Node Cards */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14, width: "100%" }}>
                      {[
                        { name: "Simple Reflex", svg: <ZapSvg size={16} color="var(--accent-strong)" />, tag: "Condition-Action", desc: "Executes direct IF/THEN rules based purely on current input. Operates without historical memory context." },
                        { name: "Model-based Reflex", svg: <BrainSvg size={16} color="var(--accent-strong)" />, tag: "Stateful Memory", desc: "Maintains internal state memory to track environment history and world state changes over time." },
                        { name: "Goal-based", svg: <TargetSvg size={16} color="var(--accent-strong)" />, tag: "Action Planning", desc: "Plans multi-step action sequences dynamically to achieve specific long-term target goals." },
                        { name: "Utility-based", svg: <ChartSvg size={16} color="var(--accent-strong)" />, tag: "Utility Optimization", desc: "Evaluates trade-offs using utility scoring functions to maximize overall task success." },
                        { name: "Learning Agent", svg: <GraduationSvg size={16} color="var(--accent-strong)" />, tag: "Adaptive Feedback", desc: "Adapts and improves execution performance over time using environment feedback loops." },
                        { name: "Hierarchical Agent", svg: <CrownSvg size={16} color="var(--accent-strong)" />, tag: "Supervisor Pattern", desc: "Supervisor agent that plans, routes, and delegates sub-tasks to specialized sub-agents." },
                        { name: "Multi-agent Systems", svg: <UsersSvg size={16} color="var(--accent-strong)" />, tag: "Swarm Collaboration", desc: "Collaborative swarms of specialized autonomous agents working together in teams." },
                      ].map((item, idx) => (
                        <div
                          key={idx}
                          style={{
                            background: "var(--panel)",
                            border: "1px solid var(--border)",
                            borderRadius: 12,
                            padding: 16,
                            display: "flex",
                            flexDirection: "column",
                            gap: 8,
                            transition: "all 0.18s ease"
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = "var(--border-strong)";
                            e.currentTarget.style.transform = "translateY(-2px)";
                            e.currentTarget.style.boxShadow = "var(--shadow-md)";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = "var(--border)";
                            e.currentTarget.style.transform = "none";
                            e.currentTarget.style.boxShadow = "none";
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid var(--border)", paddingBottom: 8 }}>
                            <div style={{ fontWeight: 800, fontSize: 14, color: "var(--accent-strong)", display: "flex", alignItems: "center", gap: 8 }}>
                              {item.svg} {item.name}
                            </div>
                            <span className="mono r" style={{ fontSize: 10, background: "var(--surface)", border: "1px solid var(--border)", padding: "2px 7px", borderRadius: 6, color: "var(--muted)", fontWeight: 600 }}>{item.tag}</span>
                          </div>
                          <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.6, fontWeight: 500 }}>
                            {item.desc}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 1. Deep Architecture & Mechanics Breakdown */}
            <div className="card" style={{ marginBottom: 20, borderRadius: 16 }}>
              <div className="card-h">
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                  <BuildSvg size={16} color="var(--accent-strong)" /> <b>First-Principles Architecture & Mechanics</b>
                </span>
                <span className="mono r" style={{ opacity: 0.8 }}>Technical Deep Dive</span>
              </div>
              <div className="card-b">
                {selectedTopic.id === "genai_vs_agent" ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    {/* Side-by-Side Comparison Boxes */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14 }}>
                      {/* Box 1: Generative AI */}
                      <div style={{
                        background: "rgba(99, 102, 241, 0.08)",
                        border: "1.5px solid rgba(99, 102, 241, 0.35)",
                        borderRadius: 14,
                        padding: 16,
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                        boxShadow: "0 4px 14px rgba(99, 102, 241, 0.1)"
                      }}>
                        <div style={{ fontSize: 14, fontWeight: 800, color: "#6366f1", borderBottom: "1px solid rgba(99, 102, 241, 0.2)", paddingBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                          <span>🤖</span> Generative AI (Text Generator)
                        </div>
                        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5, color: "var(--text)", lineHeight: 1.75 }}>
                          <li><b>Role:</b> Foundation LLM predicting text/code.</li>
                          <li><b>Execution:</b> Passive 1-shot turn (Prompt → Text Output).</li>
                          <li><b>Capabilities:</b> Essay writing, code generation, summarization.</li>
                          <li><b>Limitations:</b> Sandboxed; cannot run APIs, execute code, or modify DBs.</li>
                        </ul>
                      </div>

                      {/* Box 2: AI Agent */}
                      <div style={{
                        background: "rgba(168, 85, 247, 0.08)",
                        border: "1.5px solid rgba(168, 85, 247, 0.35)",
                        borderRadius: 14,
                        padding: 16,
                        display: "flex",
                        flexDirection: "column",
                        gap: 10,
                        boxShadow: "0 4px 14px rgba(168, 85, 247, 0.1)"
                      }}>
                        <div style={{ fontSize: 14, fontWeight: 800, color: "#a855f7", borderBottom: "1px solid rgba(168, 85, 247, 0.2)", paddingBottom: 8, display: "flex", alignItems: "center", gap: 8 }}>
                          <span>⚡</span> AI Agent (Autonomous System)
                        </div>
                        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5, color: "var(--text)", lineHeight: 1.75 }}>
                          <li><b>Role:</b> LLM Cognitive Brain + Tools + Memory + Feedback Loop.</li>
                          <li><b>Execution:</b> Active multi-step goal loop (Goal → Action → Result).</li>
                          <li><b>Capabilities:</b> Stock research, DB updates, refund processing, terminal tasks.</li>
                          <li><b>Strengths:</b> Autonomously takes real-world actions across software APIs.</li>
                        </ul>
                      </div>
                    </div>

                    <div style={{
                      fontSize: 14.5, color: "var(--text)", lineHeight: 1.8,
                      background: "var(--surface)", padding: 20, borderRadius: 14, border: "1.5px solid var(--border-strong)"
                    }}>
                      {renderHighlightedText(selectedTopic.architectureOverview)}
                    </div>
                  </div>
                ) : (
                  <div style={{
                    fontSize: 14.5, color: "var(--text)", lineHeight: 1.8,
                    background: "var(--surface)", padding: 20, borderRadius: 14, border: "1.5px solid var(--border-strong)"
                  }}>
                    {renderHighlightedText(selectedTopic.architectureOverview)}
                  </div>
                )}
              </div>
            </div>

            {/* 2. Interactive Architectural Execution Flowchart */}
            <div className="card" style={{ marginBottom: 20, borderRadius: 16, border: "1.5px solid rgba(99,102,241,0.3)" }}>
              <div className="card-h">
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                  <span>⚡</span> <b>Interactive Architectural Execution Flowchart</b>
                </span>
                <span className="badge good">{selectedTopic.flowSteps.length} Sequential Steps &amp; Branching Logic</span>
              </div>
              <div className="card-b">
                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  {selectedTopic.flowSteps.map((step, idx) => {
                    const isDecisionNode = step.label.toLowerCase().includes("inference") || step.label.toLowerCase().includes("decision") || step.label.toLowerCase().includes("reasoning") || step.detail.toLowerCase().includes("decides") || step.label.toLowerCase().includes("branch");

                    return (
                      <div key={idx} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        {/* Step Card Node */}
                        <div
                          style={{
                            width: "100%",
                            display: "flex",
                            alignItems: "flex-start",
                            gap: 16,
                            background: isDecisionNode 
                              ? "linear-gradient(135deg, rgba(168,85,247,0.12), rgba(99,102,241,0.12))" 
                              : "var(--surface)",
                            border: `1.5px solid ${isDecisionNode ? "rgba(168,85,247,0.4)" : "var(--border)"}`,
                            borderRadius: 14,
                            padding: 16,
                            boxShadow: isDecisionNode ? "0 6px 20px rgba(168,85,247,0.15)" : "0 4px 14px rgba(0,0,0,0.06)",
                            transition: "all 0.22s ease",
                            position: "relative"
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = "var(--accent)";
                            e.currentTarget.style.transform = "translateY(-2px)";
                            e.currentTarget.style.boxShadow = "0 8px 24px rgba(99,102,241,0.2)";
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = isDecisionNode ? "rgba(168,85,247,0.4)" : "var(--border)";
                            e.currentTarget.style.transform = "none";
                            e.currentTarget.style.boxShadow = isDecisionNode ? "0 6px 20px rgba(168,85,247,0.15)" : "0 4px 14px rgba(0,0,0,0.06)";
                          }}
                        >
                          <div style={{
                            width: 36, height: 36, borderRadius: 10,
                            background: isDecisionNode 
                              ? "linear-gradient(135deg, #a855f7, #6366f1)" 
                              : "linear-gradient(135deg, var(--accent), #10b981)",
                            boxShadow: "0 0 14px rgba(99,102,241,0.35)",
                            color: "#fff", display: "grid", placeItems: "center",
                            fontWeight: 800, fontSize: 14, flexShrink: 0
                          }}>
                            {isDecisionNode ? "🔀" : idx + 1}
                          </div>

                          <div style={{ flex: 1 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <span style={{ fontWeight: 800, fontSize: 15, color: "var(--text)" }}>{step.label}</span>
                              {isDecisionNode && <span className="badge" style={{ background: "rgba(168,85,247,0.2)", color: "#a855f7", border: "1px solid rgba(168,85,247,0.4)", fontSize: 11, fontWeight: 700 }}>Decision Point</span>}
                            </div>
                            <div style={{ fontSize: 13.5, color: "var(--muted)", marginTop: 6, lineHeight: 1.65, fontWeight: 500 }}>{step.detail}</div>

                            {/* If decision node, render visual branching paths */}
                            {isDecisionNode && (
                              <div style={{
                                marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10,
                                background: "var(--panel)", padding: 12, borderRadius: 10, border: "1px solid var(--border)"
                              }}>
                                <div style={{ borderLeft: "3px solid #10b981", paddingLeft: 10 }}>
                                  <div style={{ fontSize: 11.5, fontWeight: 700, color: "#10b981" }}>BRANCH A: Direct Response</div>
                                  <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>LLM generates answer string directly &amp; completes request turn.</div>
                                </div>
                                <div style={{ borderLeft: "3px solid #6366f1", paddingLeft: 10 }}>
                                  <div style={{ fontSize: 11.5, fontWeight: 700, color: "#6366f1" }}>BRANCH B: Tool Invocation</div>
                                  <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>Emits tool payload → Runtime executes → Returns Observation → Loops back.</div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Animated Glowing Connector */}
                        {idx < selectedTopic.flowSteps.length - 1 && (
                          <div style={{
                            height: 24, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                            color: "var(--accent)", fontWeight: 800, fontSize: 16, opacity: 0.8, margin: "2px 0"
                          }}>
                            <div style={{ width: 2, height: 12, background: "linear-gradient(to bottom, var(--accent), #a855f7)" }} />
                            <span style={{ marginTop: -4 }}>↓</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* 3. Applications Used For */}
            <div className="card" style={{ marginBottom: 20, borderRadius: 16, border: "1.5px solid rgba(99,102,241,0.3)" }}>
              <div className="card-h" style={{ background: "linear-gradient(135deg, rgba(99,102,241,0.12), rgba(16,185,129,0.1))" }}>
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15, color: "var(--accent-strong)" }}>
                  <span>🎯</span> <b>Applications Used For</b>
                </span>
                <span className="badge good">Production Applications &amp; Industry Impact</span>
              </div>
              <div className="card-b">
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
                  {selectedTopic.applications.map((app, idx) => (
                    <div key={idx} style={{
                      background: "var(--surface)", padding: 16, borderRadius: 12, border: "1.5px solid var(--border-strong)",
                      display: "flex", flexDirection: "column", gap: 8, boxShadow: "0 4px 14px rgba(0,0,0,0.06)"
                    }}>
                      <div style={{ fontWeight: 800, fontSize: 14, color: "var(--accent-strong)", display: "flex", alignItems: "center", gap: 6, borderBottom: "1px solid var(--border)", paddingBottom: 6 }}>
                        <span>🎯</span> {app.where}
                      </div>
                      <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.6, fontWeight: 500 }}>{app.how}</div>
                      <div style={{ fontSize: 12, color: "#10b981", fontWeight: 700, marginTop: 4, display: "flex", alignItems: "center", gap: 4, background: "rgba(16,185,129,0.08)", padding: "6px 10px", borderRadius: 8, border: "1px solid rgba(16,185,129,0.2)" }}>
                        <span>📈 Impact:</span> {app.impact}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 4. Common Mistakes & Production Fixes */}
            <div className="card" style={{ marginBottom: 20, borderRadius: 16 }}>
              <div className="card-h">
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                  <span>⚠️</span> <b>Common Developer Pitfalls & Production Fixes</b>
                </span>
              </div>
              <div className="card-b">
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12 }}>
                  {selectedTopic.pitfallsAndMistakes.map((item, idx) => (
                    <div key={idx} style={{
                      background: "rgba(239, 68, 68, 0.06)", border: "1px solid rgba(239, 68, 68, 0.25)",
                      borderRadius: 12, padding: 14
                    }}>
                      <div style={{ fontWeight: 700, fontSize: 12.5, color: "#f87171", display: "flex", alignItems: "center", gap: 6 }}>
                        <span>❌ Mistake:</span> {item.mistake}
                      </div>
                      <div style={{ fontWeight: 600, fontSize: 12, color: "#34d399", marginTop: 6, display: "flex", alignItems: "center", gap: 6 }}>
                        <span>✅ Production Fix:</span> {item.fix}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>



            {/* 6. Code Implementation Sandbox */}
            <div className="card" style={{ marginBottom: 20, borderRadius: 16 }}>
              <div className="card-h">
                <span className="t" style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 15 }}>
                  <span>💻</span> <b>Runnable Code Implementation Blueprint</b>
                </span>
                <span className="mono r" style={{ opacity: 0.7 }}>Python / TypeScript syntax</span>
              </div>
              <div className="card-b">
                <div className="learn-code-block" style={{
                  background: "var(--panel-2)", border: "1.5px solid var(--border-strong)", borderRadius: 12, padding: 18, overflowX: "auto"
                }}>
                  <pre style={{ margin: 0, fontFamily: "var(--mono)", fontSize: 13.5, color: "#f1f5f9", lineHeight: 1.7, fontWeight: 500 }}>
                    {selectedTopic.codeExample}
                  </pre>
                </div>
              </div>
            </div>





            {/* Try It Live Action Banner */}
            {selectedTopic.suggestedTemplate && (
              <div style={{
                marginTop: 14,
                background: "linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.14))",
                border: "1.5px solid var(--accent)",
                boxShadow: "0 8px 24px rgba(99,102,241,0.2)",
                padding: 20,
                borderRadius: 16,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: 14
              }}>
                <div>
                  <div style={{ fontWeight: 800, fontSize: 15, color: "var(--text)", display: "flex", alignItems: "center", gap: 8 }}>
                    <span>🚀</span> Ready to build and test this concept live in the Agent Builder?
                  </div>
                  <div style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 4 }}>
                    Instantly load the <b>&quot;{TEMPLATES.find((x) => x.id === selectedTopic.suggestedTemplate)?.label}&quot;</b> starter template.
                  </div>
                </div>
                <button
                  className="btn"
                  style={{ padding: "10px 18px", fontSize: 13, fontWeight: 700, borderRadius: 10 }}
                  onClick={() => {
                    const t = TEMPLATES.find((x) => x.id === selectedTopic!.suggestedTemplate);
                    if (t) {
                      applyTemplate(t);
                      setStep("build");
                      setSelectedTopic(null);
                      setQuizOption(null);
                    }
                  }}
                >
                  Load &quot;{TEMPLATES.find((x) => x.id === selectedTopic.suggestedTemplate)?.label}&quot; Template →
                </button>
              </div>
            )}
          </div>
        )}
      </>)}
      </>}

      <div className={`modal-wrap ${showCode ? "show" : ""}`} onClick={(e) => { if (e.target === e.currentTarget) setShowCode(false); }}>
        <div className="modal"><div className="mh"><b>Agent code · {curType.label}</b><div className="r" style={{ marginLeft: "auto", display: "flex", gap: 8 }}><button className="btn ghost sm" onClick={copyCode}>{copied ? "Copied ✓" : "Copy"}</button><button className="btn sm" onClick={downloadCode}>Download</button></div><button className="x" onClick={() => setShowCode(false)}>×</button></div><div className="mb"><div className="note" style={{ marginBottom: 10 }}>Where to use it: run this Python (<code>pip install openai</code>, set <code>OPENAI_BASE_URL</code> &amp; <code>OPENAI_API_KEY</code>) · or <b>💾 Save</b> to My Projects · or <b>⬇ Export JSON</b> to load the config into your own app.</div><div className="code">{buildCode()}</div></div></div>
      </div>
    </>
  );
}
