import "server-only";
import { and, eq } from "drizzle-orm";
import { db } from "./db";
import { providers, userProviders } from "./db/schema";
import { decrypt } from "./crypto";

export type ResolvedProvider = { id: string; baseUrl: string; apiKey: string; model: string; provider: string; label: string | null; own?: boolean };

// A user's own providers. Wrapped so a missing user_providers table (before the migration
// is applied) degrades gracefully to "no personal providers" — the app then uses global ones.
async function myProviders(userId: string) {
  try {
    return await db.select().from(userProviders).where(and(eq(userProviders.userId, userId), eq(userProviders.enabled, true)));
  } catch { return []; }
}
async function myProviderById(userId: string, id: string) {
  try {
    const rows = await db.select().from(userProviders).where(and(eq(userProviders.id, id), eq(userProviders.userId, userId))).limit(1);
    return rows[0] ?? null;
  } catch { return null; }
}
const shape = (p: { id: string; provider: string; label: string | null; baseUrl: string; apiKeyEnc: string | null; defaultModel: string | null }, own: boolean): ResolvedProvider => ({
  id: p.id, provider: p.provider, label: p.label ?? null, baseUrl: p.baseUrl, apiKey: p.apiKeyEnc ? decrypt(p.apiKeyEnc) : "", model: p.defaultModel || "", own,
});

// OpenAI-compatible endpoints for every provider — single source of truth in providerCatalog.
export { PROVIDER_CATALOG } from "./providerCatalog";

// List models straight from the provider's /models endpoint.
export async function fetchModels(baseUrl: string, apiKey: string): Promise<string[]> {
  // GitHub Models: models.github.ai paths return 410 — the working OpenAI-compatible host is the
  // Azure inference endpoint, whose /models catalog uses the friendly `name` (the `id` is an azureml:// path).
  const gh = /models\.github\.ai|models\.inference\.ai\.azure\.com/i.test(baseUrl);
  const url = gh ? "https://models.inference.ai.azure.com/models" : baseUrl.replace(/\/$/, "") + "/models";
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 12000);
  try {
    const res = await fetch(url, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`Provider returned HTTP ${res.status}`);
    const j: unknown = await res.json();
    // Handle the common shapes: {data:[…]}, {models:[…]}, or a bare top-level array.
    const arr: unknown[] = Array.isArray(j)
      ? j
      : (j as { data?: unknown[]; models?: unknown[] }).data ?? (j as { models?: unknown[] }).models ?? [];
    const ids = (arr as Array<{ id?: string; name?: string }>)
      .map((m) => {
        const id = m.id || "";
        // Prefer a clean id; fall back to `name` when the id is an azureml:// registry path.
        const usable = id && !id.startsWith("azureml://") ? id : (m.name || "");
        return usable.replace(/^models\//, "");
      })
      .filter(Boolean);
    return Array.from(new Set(ids)).sort();
  } finally {
    clearTimeout(t);
  }
}

// The provider used to serve LLM calls. Prefers the user's OWN enabled provider (their key),
// then falls back to the first enabled global/admin provider. Returns a decrypted key.
export async function getActiveProvider(userId?: string): Promise<ResolvedProvider | null> {
  if (userId) {
    const mine = await myProviders(userId);
    if (mine[0]) return shape(mine[0], true);
  }
  const rows = await db.select().from(providers).where(eq(providers.enabled, true)).limit(1);
  return rows[0] ? shape(rows[0], false) : null;
}

// Fetch one provider by id (used when the user picks a specific provider in the UI).
// Checks the user's own providers first (ownership-scoped), then global providers.
export async function getProviderById(id: string, userId?: string): Promise<ResolvedProvider | null> {
  if (userId) {
    const mine = await myProviderById(userId, id);
    if (mine && mine.enabled) return shape(mine, true);
  }
  const rows = await db.select().from(providers).where(eq(providers.id, id)).limit(1);
  const p = rows[0];
  if (!p || !p.enabled) return null;
  return shape(p, false);
}

// All enabled providers for the UI selector (no keys): the user's own first, then global.
export async function getEnabledProviders(userId?: string): Promise<{ id: string; provider: string; label: string | null; defaultModel: string; own: boolean }[]> {
  const out: { id: string; provider: string; label: string | null; defaultModel: string; own: boolean }[] = [];
  if (userId) {
    const mine = await myProviders(userId);
    for (const p of mine) out.push({ id: p.id, provider: p.provider, label: p.label ?? null, defaultModel: p.defaultModel || "", own: true });
  }
  const rows = await db.select().from(providers).where(eq(providers.enabled, true));
  for (const p of rows) out.push({ id: p.id, provider: p.provider, label: p.label ?? null, defaultModel: p.defaultModel || "", own: false });
  return out;
}
