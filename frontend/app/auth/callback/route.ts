import { NextResponse } from "next/server";
import { createClient } from "@/utils/supabase/server";

/**
 * Supabase magic-link / email-confirm callback. The email link lands here
 * with `?code=…`; we exchange it for a session cookie, then redirect home.
 */
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");
  const next = searchParams.get("next") || "/dashboard";

  if (code) {
    const supabase = await createClient();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) return NextResponse.redirect(`${origin}${next}`);
  }
  return NextResponse.redirect(`${origin}/login?error=auth`);
}
