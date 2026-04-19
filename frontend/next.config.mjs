import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Pin the workspace root to this directory so Next.js doesn't latch onto a
  // stray lockfile higher up in the filesystem (it warns otherwise on Windows).
  outputFileTracingRoot: __dirname,
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      { source: "/api/backend/:path*", destination: `${apiBase}/api/v1/:path*` },
    ];
  },
};

export default nextConfig;
