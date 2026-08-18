import { defineConfig } from "vitest/config";
import path from "path";

export default defineConfig({
  // Skip the project's Tailwind v4 PostCSS pipeline — these are pure-logic tests.
  css: { postcss: { plugins: [] } },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
  resolve: {
    alias: {
      // Let us unit-test modules that guard themselves with `import "server-only"`.
      "server-only": path.resolve(__dirname, "tests/stubs/server-only.ts"),
      "@": path.resolve(__dirname, "src"),
    },
  },
});
