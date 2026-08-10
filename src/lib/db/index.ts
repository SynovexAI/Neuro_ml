import { drizzle as drizzleMysql, type MySql2Database } from "drizzle-orm/mysql2";
import { drizzle as drizzleHttp } from "drizzle-orm/tidb-serverless";
import { connect } from "@tidbcloud/serverless";
import mysql from "mysql2/promise";
import * as schema from "./schema";

// Two ways to talk to TiDB, chosen by env so the SAME code runs on both hosts:
//  • default  → mysql2 TCP pool  (works great on a persistent server like Render)
//  • http     → TiDB Cloud serverless HTTP driver, when TIDB_DRIVER=http
//    (stateless per-request — survives high concurrency on serverless hosts like Vercel
//     without exhausting TiDB's connection limit). Same DATABASE_URL, same schema/queries.
const useHttp = process.env.TIDB_DRIVER === "http" || process.env.USE_TIDB_HTTP === "1";

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
    waitForConnections: true,
  };
}

function makeDb() {
  if (useHttp) {
    if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL is not set");
    // Reuse one HTTP client across hot-reloads / warm invocations.
    const g = globalThis as unknown as { _tidbHttp?: ReturnType<typeof connect> };
    if (!g._tidbHttp) g._tidbHttp = connect({ url: process.env.DATABASE_URL });
    return drizzleHttp(g._tidbHttp, { schema });
  }
  // Reuse the pool across hot-reloads in dev (avoids exhausting connections).
  const g = globalThis as unknown as { _pool?: mysql.Pool };
  if (!g._pool) g._pool = mysql.createPool(poolConfig());
  return drizzleMysql(g._pool, { schema, mode: "default" });
}

// Both drivers expose the same Drizzle query API; the consumers were written against the
// mysql2 type, so present a single stable type regardless of which driver is active.
export const db = makeDb() as unknown as MySql2Database<typeof schema>;
export { schema };
