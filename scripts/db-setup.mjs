// One-off: create the app database on TiDB, generate local secrets, write .env.local.
// Reads TiDB URL from env var TIDB_URL (never committed). Run: TIDB_URL="..." node scripts/db-setup.mjs
import { createConnection } from "mysql2/promise";
import { writeFileSync, existsSync } from "fs";
import { randomBytes } from "crypto";

const raw = process.env.TIDB_URL;
if (!raw) { console.error("Set TIDB_URL"); process.exit(1); }
const u = new URL(raw);
const DB = "ai_workbench";
const user = decodeURIComponent(u.username);
const pass = decodeURIComponent(u.password);
const host = u.hostname, port = Number(u.port || 4000);

// 1) write .env.local if absent
const envPath = ".env.local";
if (!existsSync(envPath)) {
  const appUrl = `mysql://${u.username}:${u.password}@${host}:${port}/${DB}`;
  const env = [
    `DATABASE_URL=${appUrl}`,
    `AUTH_SECRET=${randomBytes(32).toString("hex")}`,
    `ENCRYPTION_KEY=${randomBytes(32).toString("hex")}`,
    `NODE_ENV=development`,
    ``,
  ].join("\n");
  writeFileSync(envPath, env);
  console.log("wrote .env.local (gitignored)");
} else {
  console.log(".env.local already exists — leaving it");
}

// 2) connect + create database
try {
  const conn = await createConnection({
    host, port, user, password: pass,
    ssl: { minVersion: "TLSv1.2", rejectUnauthorized: true },
    connectTimeout: 15000,
  });
  await conn.query("CREATE DATABASE IF NOT EXISTS `ai_workbench`");
  const [rows] = await conn.query("SHOW DATABASES");
  console.log("CONNECT OK. databases:", rows.map(r => Object.values(r)[0]).join(", "));
  await conn.end();
} catch (e) {
  console.error("CONNECT FAILED:", e.code || e.message);
  process.exit(2);
}
