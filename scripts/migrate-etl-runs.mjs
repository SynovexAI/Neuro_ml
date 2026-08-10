// One-off: create the etl_runs table. Reads DATABASE_URL from .env.local.
// Run: node scripts/migrate-etl-runs.mjs
import { createConnection } from "mysql2/promise";
import { readFileSync } from "fs";

try {
  const txt = readFileSync(".env.local", "utf8");
  for (const line of txt.split("\n")) { const m = line.match(/^([A-Z0-9_]+)=(.*)$/); if (m && !process.env[m[1]]) process.env[m[1]] = m[2]; }
} catch { /* no .env.local */ }

const raw = process.env.DATABASE_URL;
if (!raw) { console.error("Set DATABASE_URL"); process.exit(1); }
const u = new URL(raw);

const DDL = `CREATE TABLE IF NOT EXISTS \`etl_runs\` (
  \`id\` varchar(36) NOT NULL,
  \`user_id\` varchar(36) NOT NULL,
  \`name\` varchar(160),
  \`mode\` varchar(24),
  \`target\` varchar(200),
  \`rows_in\` int NOT NULL DEFAULT 0,
  \`rows_out\` int NOT NULL DEFAULT 0,
  \`rows_loaded\` int NOT NULL DEFAULT 0,
  \`duration_ms\` int NOT NULL DEFAULT 0,
  \`status\` varchar(16),
  \`error\` varchar(300),
  \`ts\` timestamp DEFAULT (now()),
  CONSTRAINT \`etl_runs_id\` PRIMARY KEY(\`id\`)
)`;

try {
  const conn = await createConnection({
    host: u.hostname, port: Number(u.port || 4000), user: decodeURIComponent(u.username), password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, "") || "ai_workbench", ssl: { minVersion: "TLSv1.2", rejectUnauthorized: true }, connectTimeout: 15000,
  });
  await conn.query(DDL);
  try { await conn.query("CREATE INDEX `etl_runs_user_ts_idx` ON `etl_runs` (`user_id`,`ts`)"); } catch (e) { if (!/Duplicate|already exists/i.test(e.message)) throw e; }
  const [rows] = await conn.query("SHOW TABLES LIKE 'etl_runs'");
  console.log("OK — etl_runs present:", rows.length > 0);
  await conn.end();
} catch (e) { console.error("MIGRATION FAILED:", e.code || e.message); process.exit(2); }
