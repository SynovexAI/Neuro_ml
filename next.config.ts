import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Hide the Next.js on-screen dev indicator (the little "N" badge). Dev-only UI;
  // it never appears in a production build regardless of this setting.
  devIndicators: false,
  // Document parsers are heavy Node libs — keep them out of the bundle so the
  // /api/rag/extract route requires them at runtime (avoids pdfjs bundling issues).
  serverExternalPackages: ["pdf-parse", "mammoth", "xlsx"],
};

export default nextConfig;
