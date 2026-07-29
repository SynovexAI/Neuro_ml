import "server-only";
import { eq } from "drizzle-orm";
import { db } from "./db";
import { providers } from "./db/schema";
import { decrypt } from "./crypto";

// OpenAI-compatible endpoints for every provider (Gemini via its OpenAI-compat base).
export const PROVIDER_CATALOG: Record<string, { label: string; baseUrl: string }> = {
  groq: { label: "Groq", baseUrl: "https://api.groq.com/openai/v1" },
  cerebras: { label: "Cerebras", baseUrl: "https://api.cerebras.ai/v1" },
  gemini: { label: "Google Gemini", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai" },
  ollama: { label: "Ollama (local)", baseUrl: "http://localhost:11434/v1" },
  custom: { label: "Custom (OpenAI-compatible)", baseUrl: "" },
};

// List models straight from the provider's /models endpoint.
export async function fetchModels(baseUrl: string, apiKey: string): Promise<string[]> {
  const url = baseUrl.replace(/\/$/, "") + "/models";
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 12000);
  try {
    const res = await fetch(url, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`Provider returned HTTP ${res.status}`);
    const j: unknown = await res.json();
    const arr = (j as { data?: unknown[]; models?: unknown[] }).data
      ?? (j as { models?: unknown[] }).models ?? [];
    const ids = (arr as Array<{ id?: string; name?: string }>)
      .map((m) => (m.id || m.name || "").replace(/^models\//, ""))
      .filter(Boolean);
    return Array.from(new Set(ids)).sort();
  } finally {
    clearTimeout(t);
  }
}

// The provider used to serve LLM calls (first enabled one). Returns a decrypted key.
export async function getActiveProvider(): Promise<{ baseUrl: string; apiKey: string; model: string; provider: string } | null> {
  const rows = await db.select().from(providers).where(eq(providers.enabled, true)).limit(1);
  const p = rows[0];
  if (!p) return null;
  return {
    provider: p.provider,
    baseUrl: p.baseUrl,
    apiKey: p.apiKeyEnc ? decrypt(p.apiKeyEnc) : "",
    model: p.defaultModel || "",
  };
}
