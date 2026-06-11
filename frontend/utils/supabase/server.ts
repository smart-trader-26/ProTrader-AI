import { createServerClient, type CookieOptions } from "@supabase/ssr";
import { cookies } from "next/headers";
import { SUPABASE_ENABLED, SUPABASE_URL, SUPABASE_ANON_KEY, createStubClient } from "./stub";

export async function createClient() {
  // Local-dev mode: no Supabase configured → return the no-DB stub.
  if (!SUPABASE_ENABLED) return createStubClient();

  const cookieStore = await cookies();
  return createServerClient(
    SUPABASE_URL!,
    SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet: { name: string; value: string; options: CookieOptions }[]) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          } catch {
            // Called from a Server Component — middleware refreshes cookies.
          }
        },
      },
    },
  );
}
