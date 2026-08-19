import {
  mysqlTable, varchar, text, timestamp, boolean, json, mysqlEnum, int, bigint, index,
} from "drizzle-orm/mysql-core";

// Platform users. First registered user is promoted to admin+active.
// Others self-sign-up as status='pending' until an admin approves.
export const users = mysqlTable("users", {
  id: varchar("id", { length: 36 }).primaryKey(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  name: varchar("name", { length: 120 }),
  passwordHash: varchar("password_hash", { length: 255 }).notNull(),
  role: mysqlEnum("role", ["admin", "student"]).notNull().default("student"),
  status: mysqlEnum("status", ["pending", "active", "suspended"]).notNull().default("pending"),
  // Per-user monthly LLM token budget. NULL = use the platform default.
  // Admins are always unlimited regardless of this value.
  monthlyTokenLimit: int("monthly_token_limit"),
  createdAt: timestamp("created_at").defaultNow(),
});

// Opaque session tokens stored server-side; the cookie holds only the token.
export const sessions = mysqlTable("sessions", {
  token: varchar("token", { length: 64 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  expiresAt: timestamp("expires_at").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

// LLM providers configured by the admin. API keys are AES-256-GCM encrypted.
export const providers = mysqlTable("providers", {
  id: varchar("id", { length: 36 }).primaryKey(),
  provider: varchar("provider", { length: 40 }).notNull(), // groq | cerebras | gemini | ollama | custom
  label: varchar("label", { length: 120 }),
  baseUrl: varchar("base_url", { length: 255 }).notNull(),
  apiKeyEnc: text("api_key_enc"), // encrypted; never returned to the client
  defaultModel: varchar("default_model", { length: 120 }),
  enabled: boolean("enabled").notNull().default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
});

// Per-user LLM providers (each user brings their OWN key). Separate from the admin
// `providers` table so the shared/global flow is untouched. Resolution prefers a user's
// own enabled provider, then falls back to the global ones. Keys are AES-256-GCM encrypted.
export const userProviders = mysqlTable("user_providers", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  provider: varchar("provider", { length: 40 }).notNull(),
  label: varchar("label", { length: 120 }),
  baseUrl: varchar("base_url", { length: 255 }).notNull(),
  apiKeyEnc: text("api_key_enc"),
  defaultModel: varchar("default_model", { length: 120 }),
  enabled: boolean("enabled").notNull().default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
}, (t) => [index("user_providers_user_idx").on(t.userId)]);

// Saved student/admin builds across labs (prompt, rag, agent, ml, dl, etl, compose).
export const projects = mysqlTable("projects", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  lab: varchar("lab", { length: 40 }).notNull(),
  name: varchar("name", { length: 160 }).notNull(),
  config: json("config"),
  shared: boolean("shared").notNull().default(false),
  published: boolean("published").notNull().default(false), // usable in Workroom
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
}, (t) => [index("projects_user_idx").on(t.userId)]);

// Deployments of a published agent to a channel (Telegram bot, API key, web widget).
export const channels = mysqlTable("channels", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  projectId: varchar("project_id", { length: 36 }).notNull(),
  type: varchar("type", { length: 24 }).notNull(),   // telegram | api | widget
  secretEnc: text("secret_enc"),                     // encrypted bot token / API key
  enabled: boolean("enabled").notNull().default(true),
  dailyLimit: int("daily_limit"),                    // max runs/day for public channels (NULL = default)
  createdAt: timestamp("created_at").defaultNow(),
}, (t) => [index("channels_project_idx").on(t.projectId)]);

// Every LLM call is metered here (one row per completion). Powers per-user
// quota enforcement and admin usage reporting.
export const usage = mysqlTable("usage", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  lab: varchar("lab", { length: 40 }),
  model: varchar("model", { length: 120 }),
  promptTokens: int("prompt_tokens").notNull().default(0),
  completionTokens: int("completion_tokens").notNull().default(0),
  totalTokens: int("total_tokens").notNull().default(0),
  // True when the provider didn't report usage and we estimated from text length.
  estimated: boolean("estimated").notNull().default(false),
  ts: timestamp("ts").defaultNow(),
}, (t) => [index("usage_user_ts_idx").on(t.userId, t.ts), index("usage_ts_idx").on(t.ts)]);

// Persistent fixed-window rate limiter (survives restarts / works across
// serverless instances, unlike the in-memory fallback). One row per
// (scope, user, window). id = `${scope}:${userId}:${windowStart}`.
export const rateLimits = mysqlTable("rate_limits", {
  id: varchar("id", { length: 140 }).primaryKey(),
  scope: varchar("scope", { length: 40 }).notNull(),
  windowStart: bigint("window_start", { mode: "number" }).notNull(),
  count: int("count").notNull().default(0),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
});

// Lightweight audit / event log for monitoring (logins, quota hits, admin actions).
export const auditLog = mysqlTable("audit_log", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }),
  event: varchar("event", { length: 60 }).notNull(),
  detail: json("detail"),
  ts: timestamp("ts").defaultNow(),
}, (t) => [index("audit_event_ts_idx").on(t.event, t.ts)]);

// Admin-connected MCP (Model Context Protocol) servers. Their tools become
// available to agents. Secrets (API keys / tokens) are AES-256-GCM encrypted.
export const mcpServers = mysqlTable("mcp_servers", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }),  // owner; each user has their own MCP servers
  name: varchar("name", { length: 80 }).notNull(),
  transport: mysqlEnum("transport", ["http", "sse", "stdio"]).notNull().default("http"),
  url: varchar("url", { length: 500 }),        // http / sse
  command: varchar("command", { length: 500 }), // stdio
  authType: mysqlEnum("auth_type", ["none", "apikey", "bearer", "oauth"]).notNull().default("none"),
  headerName: varchar("header_name", { length: 80 }), // e.g. Authorization or X-API-Key
  envName: varchar("env_name", { length: 80 }),        // stdio: env var the secret is passed as
  secretEnc: text("secret_enc"),               // encrypted token / key; never returned to the client
  enabled: boolean("enabled").notNull().default(true),
  tools: json("tools"),                        // discovered tool names (populated later)
  status: varchar("status", { length: 40 }).default("configured"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
}, (t) => [index("mcp_user_idx").on(t.userId)]);

// One row per agent run (in-browser ReAct/workflow or NAT). Powers the agent
// analytics: success rate, iterations, tool usage, tokens/cost, latency.
export const agentRuns = mysqlTable("agent_runs", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  agentName: varchar("agent_name", { length: 120 }),
  agentType: varchar("agent_type", { length: 24 }),   // react | workflow | nat
  runtime: varchar("runtime", { length: 16 }),        // browser | nat
  provider: varchar("provider", { length: 40 }),
  model: varchar("model", { length: 120 }),
  iterations: int("iterations").notNull().default(0),
  toolCalls: json("tool_calls"),                      // [{ tool, count }]
  toolCallCount: int("tool_call_count").notNull().default(0),
  promptTokens: int("prompt_tokens").notNull().default(0),
  completionTokens: int("completion_tokens").notNull().default(0),
  totalTokens: int("total_tokens").notNull().default(0),
  latencyMs: int("latency_ms").notNull().default(0),
  outcome: varchar("outcome", { length: 24 }),        // success | max_iters | error | stopped
  errorMsg: varchar("error_msg", { length: 300 }),
  ts: timestamp("ts").defaultNow(),
}, (t) => [index("agent_runs_user_ts_idx").on(t.userId, t.ts)]);

// Per-user reusable knowledge bases (RAG). Documents are chunked + embedded on
// sync; vectors are stored in kb_chunks (dense embeddings when the provider
// supports them, else TF-IDF sparse vectors). Agents select a KB to ground on.
export const knowledgeBases = mysqlTable("knowledge_bases", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  name: varchar("name", { length: 120 }).notNull(),
  status: varchar("status", { length: 24 }).notNull().default("empty"), // empty | syncing | ready | error
  docCount: int("doc_count").notNull().default(0),
  chunkCount: int("chunk_count").notNull().default(0),
  embModel: varchar("emb_model", { length: 120 }),  // embedding model name, or "tfidf"
  embMeta: json("emb_meta"),                        // { df, N } for TF-IDF query reconstruction
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
}, (t) => [index("kb_user_idx").on(t.userId)]);

// Exact original document text per KB (so KB → RAG/agent uses the real text, not a
// chunk-reconstruction). Populated on sync alongside kb_chunks.
export const kbDocs = mysqlTable("kb_docs", {
  id: varchar("id", { length: 36 }).primaryKey(),
  kbId: varchar("kb_id", { length: 36 }).notNull(),
  name: varchar("name", { length: 200 }),
  text: text("text"),
}, (t) => [index("kb_docs_kb_idx").on(t.kbId)]);

export const kbChunks = mysqlTable("kb_chunks", {
  id: varchar("id", { length: 36 }).primaryKey(),
  kbId: varchar("kb_id", { length: 36 }).notNull(),
  docName: varchar("doc_name", { length: 200 }),
  idx: int("idx").notNull().default(0),
  text: text("text"),
  embedding: json("embedding"),  // dense float[] (embeddings) or sparse {term:weight} (TF-IDF)
}, (t) => [index("kb_chunks_kb_idx").on(t.kbId)]);

export type User = typeof users.$inferSelect;
export type Provider = typeof providers.$inferSelect;
export type UserProvider = typeof userProviders.$inferSelect;
export type Project = typeof projects.$inferSelect;
export type Usage = typeof usage.$inferSelect;
export type McpServer = typeof mcpServers.$inferSelect;
export type AgentRun = typeof agentRuns.$inferSelect;
// ETL "Load to database" target: a real persisted output of a pipeline run.
// The rows are stored as JSON in etl_dataset_rows, scoped to the user.
export const etlDatasets = mysqlTable("etl_datasets", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  name: varchar("name", { length: 160 }).notNull(),
  cols: json("cols"),                       // string[]
  rowCount: int("row_count").notNull().default(0),
  createdAt: timestamp("created_at").defaultNow(),
}, (t) => [index("etl_ds_user_idx").on(t.userId)]);

export const etlDatasetRows = mysqlTable("etl_dataset_rows", {
  id: varchar("id", { length: 36 }).primaryKey(),
  datasetId: varchar("dataset_id", { length: 36 }).notNull(),
  idx: int("idx").notNull().default(0),
  data: json("data"),                       // one record
}, (t) => [index("etl_rows_ds_idx").on(t.datasetId)]);

export type KnowledgeBase = typeof knowledgeBases.$inferSelect;
export type KbChunk = typeof kbChunks.$inferSelect;
export type Channel = typeof channels.$inferSelect;
export type EtlDataset = typeof etlDatasets.$inferSelect;
