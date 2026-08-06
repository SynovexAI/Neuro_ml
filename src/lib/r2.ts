import "server-only";
import { AwsClient } from "aws4fetch";

// Cloudflare R2 object storage (S3-compatible) for uploaded files / artifacts.
// Configured entirely by env — if unset, `r2Configured()` is false and callers fall back
// to their old behaviour, so the app keeps working with no R2 account.
//
// Env vars (see the setup guide):
//   R2_ACCOUNT_ID          your Cloudflare account id
//   R2_ACCESS_KEY_ID       R2 API token access key id
//   R2_SECRET_ACCESS_KEY   R2 API token secret
//   R2_BUCKET              the bucket name
//   R2_PUBLIC_URL          (optional) public/custom-domain base for public reads, e.g. https://files.example.com
const ACCOUNT = process.env.R2_ACCOUNT_ID || "";
const ACCESS = process.env.R2_ACCESS_KEY_ID || "";
const SECRET = process.env.R2_SECRET_ACCESS_KEY || "";
const BUCKET = process.env.R2_BUCKET || "";
const PUBLIC_BASE = (process.env.R2_PUBLIC_URL || "").replace(/\/$/, "");

export function r2Configured(): boolean {
  return !!(ACCOUNT && ACCESS && SECRET && BUCKET);
}

const endpoint = () => `https://${ACCOUNT}.r2.cloudflarestorage.com`;
const objectUrl = (key: string) => `${endpoint()}/${BUCKET}/${key.split("/").map(encodeURIComponent).join("/")}`;

let _client: AwsClient | null = null;
function client(): AwsClient {
  if (!_client) _client = new AwsClient({ accessKeyId: ACCESS, secretAccessKey: SECRET, region: "auto", service: "s3" });
  return _client;
}

// Upload bytes to R2. Returns the object key and a URL (public if R2_PUBLIC_URL is set,
// otherwise a short-lived signed GET URL).
export async function putObject(key: string, body: Uint8Array | ArrayBuffer | Buffer, contentType = "application/octet-stream"): Promise<{ key: string; url: string }> {
  if (!r2Configured()) throw new Error("R2 is not configured (set R2_* env vars).");
  const res = await client().fetch(objectUrl(key), { method: "PUT", body: body as BodyInit, headers: { "content-type": contentType } });
  if (!res.ok) throw new Error(`R2 upload failed: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
  const url = PUBLIC_BASE ? `${PUBLIC_BASE}/${key}` : await signedGetUrl(key);
  return { key, url };
}

// Fetch an object's bytes back from R2.
export async function getObject(key: string): Promise<ArrayBuffer | null> {
  if (!r2Configured()) return null;
  const res = await client().fetch(objectUrl(key), { method: "GET" });
  if (!res.ok) return null;
  return res.arrayBuffer();
}

// A time-limited signed GET URL (works even for private buckets). Default 1 hour.
export async function signedGetUrl(key: string, expiresSeconds = 3600): Promise<string> {
  if (!r2Configured()) throw new Error("R2 is not configured.");
  const url = new URL(objectUrl(key));
  url.searchParams.set("X-Amz-Expires", String(expiresSeconds));
  const signed = await client().sign(url.toString(), { method: "GET", aws: { signQuery: true } });
  return signed.url;
}

export async function deleteObject(key: string): Promise<void> {
  if (!r2Configured()) return;
  await client().fetch(objectUrl(key), { method: "DELETE" });
}
