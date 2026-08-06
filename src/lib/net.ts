import "server-only";
import { lookup } from "dns/promises";

// SSRF guard — block loopback / private / link-local (cloud-metadata) targets.
export function isBlockedHost(host: string): boolean {
  const h = host.toLowerCase().replace(/^\[|\]$/g, "");
  if (h === "localhost" || h === "::1" || h === "0.0.0.0" || h.endsWith(".internal") || h.endsWith(".local")) return true;
  if (/^127\./.test(h) || /^10\./.test(h) || /^192\.168\./.test(h) || /^169\.254\./.test(h) || /^172\.(1[6-9]|2\d|3[01])\./.test(h)) return true;
  if (h.startsWith("fc") || h.startsWith("fd") || h.startsWith("fe80")) return true; // IPv6 ULA / link-local
  return false;
}

// Resolves the hostname and blocks if it points at a private/loopback address
// (defends against DNS names that resolve to internal IPs). Fails-open only on
// resolution errors so legitimate public hosts still work.
export async function resolvesToPrivate(host: string): Promise<boolean> {
  if (isBlockedHost(host)) return true;
  try { const { address } = await lookup(host); return isBlockedHost(address); }
  catch { return false; }
}

// Simple in-memory fixed-window rate limiter keyed by (user + route). Returns
// true if the request is allowed. Best-effort (per server instance).
const buckets = new Map<string, { count: number; reset: number }>();
export function rateLimit(key: string, max: number, windowMs: number): boolean {
  const now = Date.now();
  const b = buckets.get(key);
  if (!b || now > b.reset) { buckets.set(key, { count: 1, reset: now + windowMs }); return true; }
  if (b.count >= max) return false;
  b.count++;
  return true;
}
