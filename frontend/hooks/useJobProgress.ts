/**
 * useJobProgress — SSE-style progress for long-running jobs (B4.6).
 *
 * Replaces raw `waitForJob` polling with a hook that provides:
 *  - `phase`: human-readable status string (e.g., "fitting XGB", "stacking")
 *  - `progress`: 0–100 percentage (estimated from elapsed time vs typical duration)
 *  - `result`: the final payload when done
 *  - `error`: error message if the job failed
 *
 * Under the hood this still polls `/api/v1/jobs/{id}` (the backend doesn't
 * implement SSE yet), but the hook encapsulates the timing and progress
 * estimation so components can render a smooth progress bar.
 *
 * When the backend adds true SSE (EventSource), only this hook needs to
 * change — components stay the same.
 *
 * Usage:
 *   const { phase, progress, result, error, isRunning } = useJobProgress<PredictionBundle>(jobId);
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiGet } from "@/lib/api-client";

export interface JobProgressResult<T> {
  phase: string;
  progress: number; // 0–100
  result: T | null;
  error: string | null;
  isRunning: boolean;
}

interface JobPollResponse<T> {
  status?: string;
  state?: string;
  result?: T;
  error?: string;
}

const POLL_INTERVAL_MS = 2000;
const TYPICAL_DURATION_MS = 60_000; // 60s is typical for a predict job

export function useJobProgress<T>(
  jobId: string | null,
  opts: { timeoutMs?: number; typicalDurationMs?: number } = {},
): JobProgressResult<T> {
  const { timeoutMs = 180_000, typicalDurationMs = TYPICAL_DURATION_MS } = opts;
  const [phase, setPhase] = useState("queued");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const startRef = useRef<number>(0);

  const poll = useCallback(async () => {
    if (!jobId) return;
    setIsRunning(true);
    startRef.current = Date.now();

    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));

      try {
        const job = await apiGet<JobPollResponse<T>>(`/api/v1/jobs/${jobId}`);
        const state = job.status ?? job.state ?? "running";
        setPhase(state);

        if (state === "succeeded") {
          setProgress(100);
          setResult(job.result as T);
          setIsRunning(false);
          return;
        }

        if (state === "failed") {
          setProgress(0);
          setError(job.error ?? "Job failed");
          setIsRunning(false);
          return;
        }

        // Estimate progress from elapsed time
        const elapsed = Date.now() - startRef.current;
        const pct = Math.min(95, Math.round((elapsed / typicalDurationMs) * 100));
        setProgress(pct);
      } catch (e) {
        // Network error during poll — keep trying
        continue;
      }
    }

    setError("Timed out waiting for job");
    setIsRunning(false);
  }, [jobId, timeoutMs, typicalDurationMs]);

  useEffect(() => {
    if (jobId) {
      setResult(null);
      setError(null);
      setPhase("queued");
      setProgress(0);
      poll();
    }
  }, [jobId, poll]);

  return { phase, progress, result, error, isRunning };
}
