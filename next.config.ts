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
  serverExternalPackages: ["pdf-parse", "mammoth", "xlsx"],
  webpack: (config) => {
    // alasql (ELT SQL engine) optionally requires React-Native / fs modules for
    // its Node file APIs. We only run in-memory SQL, so stub them out to keep the
    // browser bundle clean.
    config.resolve.alias = {
      ...config.resolve.alias,
      "react-native-fs": false,
      "react-native-fetch-blob": false,
      "react-native": false,
    };
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false };
    return config;
  },
};

export default nextConfig;
