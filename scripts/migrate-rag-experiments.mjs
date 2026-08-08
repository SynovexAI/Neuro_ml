// One-off: create the rag_experiments table on the app DB.
// Reads DATABASE_URL from .env.local. Run: node scripts/migrate-rag-experiments.mjs
import { createConnection } from "mysql2/promise";
import { readFileSync } from "fs";

function loadEnv() {
  try {
    const txt = readFileSync(".env.local", "utf8");
    for (const line of txt.split("\n")) {
      const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
    }
  } catch { /* no .env.local */ }
}
loadEnv();

const raw = process.env.DATABASE_URL;
if (!raw) { console.error("Set DATABASE_URL (or add it to .env.local)"); process.exit(1); }
const u = new URL(raw);

const DDL_TABLE = `CREATE TABLE IF NOT EXISTS \`rag_experiments\` (
  \`id\` varchar(36) NOT NULL,
  \`user_id\` varchar(36) NOT NULL,
  \`label\` varchar(160) NOT NULL,
  \`dataset\` varchar(200),
  \`question\` text,
  \`config\` json,
  \`metrics\` json,
  \`chunk_count\` int NOT NULL DEFAULT 0,
  \`latency_ms\` int NOT NULL DEFAULT 0,
  \`ts\` timestamp DEFAULT (now()),
  CONSTRAINT \`rag_experiments_id\` PRIMARY KEY(\`id\`)
)`;

try {
  const conn = await createConnection({
    host: u.hostname, port: Number(u.port || 4000),
    user: decodeURIComponent(u.username), password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, "") || "ai_workbench",
    ssl: { minVersion: "TLSv1.2", rejectUnauthorized: true }, connectTimeout: 15000,
  });
  await conn.query(DDL_TABLE);
  // Add index if missing (TiDB has no CREATE INDEX IF NOT EXISTS on all versions → guard).
  try { await conn.query("CREATE INDEX `rag_exp_user_ts_idx` ON `rag_experiments` (`user_id`,`ts`)"); }
  catch (e) { if (!/Duplicate|already exists/i.test(e.message)) throw e; }
  const [rows] = await conn.query("SHOW TABLES LIKE 'rag_experiments'");
  console.log("OK — rag_experiments present:", rows.length > 0);
  await conn.end();
} catch (e) {
  console.error("MIGRATION FAILED:", e.code || e.message);
  process.exit(2);
}
