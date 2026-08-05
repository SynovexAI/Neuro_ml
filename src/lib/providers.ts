import "server-only";
import { eq } from "drizzle-orm";
import { db } from "./db";
import { providers } from "./db/schema";
import { decrypt } from "./crypto";

// OpenAI-compatible endpoints for every provider — single source of truth in providerCatalog.
export { PROVIDER_CATALOG } from "./providerCatalog";

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
export async function getActiveProvider(): Promise<{ id: string; baseUrl: string; apiKey: string; model: string; provider: string; label: string | null } | null> {
  const rows = await db.select().from(providers).where(eq(providers.enabled, true)).limit(1);
  const p = rows[0];
  if (!p) return null;
  return {
    id: p.id,
    provider: p.provider,
    label: p.label ?? null,
    baseUrl: p.baseUrl,
    apiKey: p.apiKeyEnc ? decrypt(p.apiKeyEnc) : "",
    model: p.defaultModel || "",
  };
}

// Fetch one enabled provider by id (used when the user picks a specific provider in the UI).
export async function getProviderById(id: string): Promise<{ id: string; baseUrl: string; apiKey: string; model: string; provider: string; label: string | null } | null> {
  const rows = await db.select().from(providers).where(eq(providers.id, id)).limit(1);
  const p = rows[0];
  if (!p || !p.enabled) return null;
  return {
    id: p.id,
    provider: p.provider,
    label: p.label ?? null,
    baseUrl: p.baseUrl,
    apiKey: p.apiKeyEnc ? decrypt(p.apiKeyEnc) : "",
    model: p.defaultModel || "",
  };
}

// All enabled providers (id + label for the UI selector, no keys).
export async function getEnabledProviders(): Promise<{ id: string; provider: string; label: string | null; defaultModel: string }[]> {
  const rows = await db.select().from(providers).where(eq(providers.enabled, true));
  return rows.map((p) => ({ id: p.id, provider: p.provider, label: p.label ?? null, defaultModel: p.defaultModel || "" }));
}
