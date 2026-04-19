/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      { source: "/api/backend/:path*", destination: `${apiBase}/api/v1/:path*` },
    ];
  },
};

export default nextConfig;
