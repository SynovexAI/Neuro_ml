import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Hide the Next.js on-screen dev indicator (the little "N" badge). Dev-only UI;
  // it never appears in a production build regardless of this setting.
  devIndicators: false,
  // Lets a verification build write to an isolated dir (BUILD_DIST_DIR) so it
  // never clobbers the running dev server's .next.
  ...(process.env.BUILD_DIST_DIR ? { distDir: process.env.BUILD_DIST_DIR } : {}),
  // Document parsers are heavy Node libs — keep them out of the bundle so the
  // /api/rag/extract route requires them at runtime (avoids pdfjs bundling issues).
  serverExternalPackages: ["pdf-parse", "mammoth", "xlsx", "@dsnp/parquetjs"],
  webpack: (config, { dev }) => {
    // Use in‑memory cache during development to avoid filesystem rename errors
    config.cache = dev ? { type: 'memory' } : { type: 'filesystem' };
    // Keep existing alias and fallback settings
    config.resolve = config.resolve || ({} as any);
    config.resolve.alias = Object.assign({}, config.resolve.alias || {}, {
      "react-native-fs": false,
      "react-native-fetch-blob": false,
      "react-native": false,
    });
    config.resolve.fallback = Object.assign({}, config.resolve.fallback || {}, { fs: false, path: false });
    return config;
  },

};

export default nextConfig;
