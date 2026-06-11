import { createBrowserClient } from "@supabase/ssr";
import { SUPABASE_ENABLED, SUPABASE_URL, SUPABASE_ANON_KEY, createStubClient } from "./stub";

export function createClient() {
  // Local-dev mode: no Supabase configured → return the no-DB stub so the
  // browser never tries to reach a Supabase project that doesn't exist.
  if (!SUPABASE_ENABLED) return createStubClient();
  return createBrowserClient(SUPABASE_URL!, SUPABASE_ANON_KEY!);
}
