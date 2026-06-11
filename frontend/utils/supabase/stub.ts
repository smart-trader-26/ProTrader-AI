/**
 * LOCAL-DEV MODE (no database / no auth)
 * ======================================
 * When the `NEXT_PUBLIC_SUPABASE_*` env vars are absent, the app runs entirely
 * locally with NO Supabase and NO database — every request behaves as a single
 * signed-in local user. This lets you test the predictor on your machine without
 * standing up Supabase. Set the env vars (see `.env.local.example`) to switch
 * real auth + persistence back on automatically; nothing here is hard-removed.
 *
 * The stub below is a minimal, type-loose Supabase-client shim that:
 *   • reports a fake signed-in user (so middleware/pages don't redirect to /login)
 *   • returns empty results for any `.from(...).select()...` DB query (no crash)
 *   • no-ops auth mutations (signIn / signUp / signOut / OAuth callback)
 */

export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL;
export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

/** True only when real Supabase credentials are configured. */
export const SUPABASE_ENABLED = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

/** Fake user used in local-dev mode so the UI renders the "signed-in" state. */
export const LOCAL_DEV_USER = {
  id: "local-dev-user",
  email: "local@dev",
  user_metadata: { full_name: "Local Dev" },
  app_metadata: { provider: "local" },
  aud: "authenticated",
  created_at: new Date(0).toISOString(),
} as const;

/** A chainable, awaitable proxy that resolves to an empty Postgres result. */
function emptyQuery(): unknown {
  const proxy: unknown = new Proxy(function () {} as unknown as object, {
    get(_t, prop) {
      if (prop === "then") {
        return (resolve: (v: unknown) => unknown) =>
          resolve({ data: [], error: null, count: 0, status: 200, statusText: "OK" });
      }
      // Any chained method (.select / .eq / .order / .single / .insert …) → same proxy.
      return () => proxy;
    },
    apply() {
      return proxy;
    },
  });
  return proxy;
}

/**
 * Minimal Supabase-client stub for local-dev mode. Typed loosely on purpose —
 * callers only use `auth.getUser/getSession/signOut/...` and `from(...)`.
 */
export function createStubClient(): ReturnType<typeof makeStub> {
  return makeStub();
}

function makeStub() {
  const ok = { error: null };
  const auth = {
    getUser: async () => ({ data: { user: LOCAL_DEV_USER }, error: null }),
    getSession: async () => ({ data: { session: null }, error: null }),
    signOut: async () => ok,
    signInWithPassword: async () => ({
      data: { user: LOCAL_DEV_USER, session: null },
      error: null,
    }),
    signUp: async () => ({ data: { user: LOCAL_DEV_USER, session: null }, error: null }),
    signInWithOAuth: async () => ({ data: { provider: "local", url: "/" }, error: null }),
    exchangeCodeForSession: async () => ({ data: { session: null }, error: null }),
    onAuthStateChange: () => ({ data: { subscription: { unsubscribe() {} } } }),
  };
  // Typed loosely (any): callers only use `.auth.*` and `.from(...)`, and the
  // real SupabaseClient surface is far larger than this local-mode shim needs.
  return {
    auth,
    from: () => emptyQuery(),
    rpc: () => emptyQuery(),
  } as any;
}
