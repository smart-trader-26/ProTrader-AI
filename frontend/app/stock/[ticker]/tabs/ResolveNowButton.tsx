"use client";

import { useState } from "react";
import { apiPost } from "@/lib/api-client";

interface ResolveResponse {
  resolved: number;
  message: string;
}

export default function ResolveNowButton() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResolveResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleResolve() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // Use apiPost so the request goes to http://localhost:8000/api/v1/...
      // (same pattern as every other client-side API call in the app).
      const data = await apiPost<ResolveResponse>("/api/v1/accuracy/resolve", {});
      setResult(data);
      // Refresh the page after 2s to show updated accuracy data
      if (data.resolved > 0) {
        setTimeout(() => window.location.reload(), 2000);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={handleResolve}
        disabled={loading}
        className="btn btn-primary text-sm"
      >
        {loading ? "Resolving…" : "Resolve predictions now"}
      </button>
      {result && (
        <p className={`text-xs ${result.resolved > 0 ? "text-bull" : "text-muted"}`}>
          {result.message}
        </p>
      )}
      {error && <p className="text-xs text-bear">{error}</p>}
    </div>
  );
}
