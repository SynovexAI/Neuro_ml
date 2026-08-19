// One-off: add channels.config_enc (encrypted per-channel config for ETL deploy).
// Reads DATABASE_URL from .env.local. Run: node scripts/migrate-channels-configenc.mjs
import { createConnection } from "mysql2/promise";
import { readFileSync } from "fs";

try {
  const txt = readFileSync(".env.local", "utf8");
  for (const line of txt.split("\n")) { const m = line.match(/^([A-Z0-9_]+)=(.*)$/); if (m && !process.env[m[1]]) process.env[m[1]] = m[2]; }
} catch { /* no .env.local */ }

const raw = process.env.DATABASE_URL;
if (!raw) { console.error("Set DATABASE_URL (or add it to .env.local)"); process.exit(1); }
const u = new URL(raw);

try {
  const conn = await createConnection({
    host: u.hostname, port: Number(u.port || 4000),
    user: decodeURIComponent(u.username), password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, "") || "ai_workbench",
    ssl: { minVersion: "TLSv1.2", rejectUnauthorized: true }, connectTimeout: 15000,
  });
  try { await conn.query("ALTER TABLE `channels` ADD COLUMN `config_enc` text"); console.log("OK — added channels.config_enc"); }
  catch (e) { if (/Duplicate column|already exists/i.test(e.message)) console.log("OK — channels.config_enc already exists"); else throw e; }
  await conn.end();
} catch (e) {
  console.error("MIGRATION FAILED:", e.code || e.message);
  process.exit(2);
}
