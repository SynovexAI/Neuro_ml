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
  };
}

// Reuse the pool across hot-reloads / serverless invocations.
const g = globalThis as unknown as { _pool?: mysql.Pool };
const pool = g._pool ?? mysql.createPool(poolConfig());
if (process.env.NODE_ENV !== "production") g._pool = pool;

export const db = drizzle(pool, { schema, mode: "default" });
export { schema };
