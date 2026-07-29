import { defineConfig } from "drizzle-kit";
import { readFileSync } from "fs";

// drizzle-kit CLI doesn't auto-load .env.local — parse it here.
try {
  readFileSync(".env.local", "utf8").split("\n").forEach((line) => {
    const i = line.indexOf("=");
    if (i > 0) {
      const k = line.slice(0, i).trim();
      const v = line.slice(i + 1).trim();
      if (k && process.env[k] === undefined) process.env[k] = v;
    }
  });
} catch { /* ignore */ }

const u = new URL(process.env.DATABASE_URL!);

export default defineConfig({
  schema: "./src/lib/db/schema.ts",
  out: "./drizzle",
  dialect: "mysql",
  dbCredentials: {
    host: u.hostname,
    port: Number(u.port || 4000),
    user: decodeURIComponent(u.username),
    password: decodeURIComponent(u.password),
    database: u.pathname.replace(/^\//, ""),
    ssl: { minVersion: "TLSv1.2", rejectUnauthorized: true },
  },
});
