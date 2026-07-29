import {
  mysqlTable, varchar, text, timestamp, boolean, json, mysqlEnum,
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

export type User = typeof users.$inferSelect;
export type Provider = typeof providers.$inferSelect;
export type Project = typeof projects.$inferSelect;
