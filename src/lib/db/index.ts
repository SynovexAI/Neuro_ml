import { drizzle } from "drizzle-orm/mysql2";
import mysql from "mysql2/promise";
import * as schema from "./schema";

// Parse DATABASE_URL and force TLS (required by TiDB Cloud serverless).
function poolConfig(): mysql.PoolOptions {
  const url = process.env.DATABASE_URL;
  if (!url) throw new Error("DATABASE_URL is not set");
  const u = new URL(url);
  return {
    host: u.hostname,
    port: Number(u.port || 4000),
    user: decodeURIComponent(u.username),
    password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, ""),
    ssl: { minVersion: "TLSv1.2", rejectUnauthorized: true },
    connectionLimit: 5,
    // Keep TCP connections alive so TiDB Cloud doesn't silently close idle ones.
    enableKeepAlive: true,
    keepAliveInitialDelay: 10000,
    // Timeout for establishing a new connection.
    connectTimeout: 15000,
    // Hard timeout per query — ensures a stale connection fails fast (5 s)
    // instead of hanging indefinitely until the OS TCP timeout fires.
    queryTimeout: 5000,
    waitForConnections: true,
  };
}

// Reuse the pool across hot-reloads in dev (avoids exhausting connections).
// In production a fresh pool per cold start is fine.
const g = globalThis as unknown as { _pool?: mysql.Pool };
if (!g._pool) g._pool = mysql.createPool(poolConfig());
const pool = g._pool;

export const db = drizzle(pool, { schema, mode: "default" });
export { schema };
