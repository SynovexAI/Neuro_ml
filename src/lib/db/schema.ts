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

// Saved student/admin builds across labs (prompt, rag, agent, ml, dl, etl, compose).
export const projects = mysqlTable("projects", {
  id: varchar("id", { length: 36 }).primaryKey(),
  userId: varchar("user_id", { length: 36 }).notNull(),
  lab: varchar("lab", { length: 40 }).notNull(),
  name: varchar("name", { length: 160 }).notNull(),
  config: json("config"),
  shared: boolean("shared").notNull().default(false),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow().onUpdateNow(),
});

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
}, (t) => [index("usage_user_ts_idx").on(t.userId, t.ts)]);

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
});

export type User = typeof users.$inferSelect;
export type Provider = typeof providers.$inferSelect;
export type Project = typeof projects.$inferSelect;
export type Usage = typeof usage.$inferSelect;
export type McpServer = typeof mcpServers.$inferSelect;
