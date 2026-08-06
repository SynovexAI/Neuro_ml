import { createCipheriv, createDecipheriv, randomBytes } from "crypto";

// AES-256-GCM. ENCRYPTION_KEY must be 64 hex chars (32 bytes).
function key(): Buffer {
  const hex = process.env.ENCRYPTION_KEY || "";
  if (hex.length !== 64) throw new Error("ENCRYPTION_KEY must be 32 bytes (64 hex chars)");
  return Buffer.from(hex, "hex");
}

export function encrypt(plain: string): string {
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", key(), iv);
  const enc = Buffer.concat([c.update(plain, "utf8"), c.final()]);
  const tag = c.getAuthTag();
  return Buffer.concat([iv, tag, enc]).toString("base64");
}

export function decrypt(b64: string): string {
  const buf = Buffer.from(b64, "base64");
  const iv = buf.subarray(0, 12);
  const tag = buf.subarray(12, 28);
  const enc = buf.subarray(28);
  const d = createDecipheriv("aes-256-gcm", key(), iv);
  d.setAuthTag(tag);
  return Buffer.concat([d.update(enc), d.final()]).toString("utf8");
}

// Show only the last 4 chars of a key to the admin UI.
export function maskKey(plain: string): string {
  if (!plain) return "";
  return "••••••••" + plain.slice(-4);
}
