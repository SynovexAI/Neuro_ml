import "server-only";
import { r2Configured, putObject as r2Put, deleteObject as r2Del } from "./r2";

// Storage abstraction: Vercel Blob (primary, no card) → Cloudflare R2 (fallback if configured).
// If neither is set, storageConfigured() is false and callers fall back to their old behaviour.
const hasBlob = () => !!process.env.BLOB_READ_WRITE_TOKEN;

export function storageConfigured(): boolean {
  return hasBlob() || r2Configured();
}
export function storageBackend(): "blob" | "r2" | null {
  if (hasBlob()) return "blob";
  if (r2Configured()) return "r2";
  return null;
}

export type StoredFile = { key: string; url: string; size: number; uploadedAt?: string };

export async function putFile(key: string, body: Buffer | Uint8Array, contentType = "application/octet-stream"): Promise<{ key: string; url: string }> {
  if (hasBlob()) {
    const { put } = await import("@vercel/blob");
    const res = await put(key, body as Buffer, { access: "public", contentType, addRandomSuffix: false });
    return { key: res.pathname, url: res.url };
  }
  if (r2Configured()) return r2Put(key, body, contentType);
  throw new Error("No storage configured (set BLOB_READ_WRITE_TOKEN, or the R2_* vars).");
}

// List stored files (for the admin storage manager). Blob only; R2 listing isn't wired here.
export async function listFiles(): Promise<StoredFile[]> {
  if (!hasBlob()) return [];
  const { list } = await import("@vercel/blob");
  const out: StoredFile[] = [];
  let cursor: string | undefined;
  do {
    const r = await list({ cursor, limit: 1000 });
    for (const b of r.blobs) out.push({ key: b.pathname, url: b.url, size: b.size, uploadedAt: b.uploadedAt ? new Date(b.uploadedAt).toISOString() : undefined });
    cursor = r.cursor;
  } while (cursor && out.length < 5000);
  return out;
}

// Delete a file. Blob wants the full URL; R2 wants the key.
export async function deleteFile(keyOrUrl: string): Promise<void> {
  if (hasBlob()) { const { del } = await import("@vercel/blob"); await del(keyOrUrl); return; }
  if (r2Configured()) await r2Del(keyOrUrl);
}
