import type { Metadata } from "next";
import "./globals.css";
import { createClient } from "@/utils/supabase/server";
import Link from "next/link";
import SignOutButton from "@/components/SignOutButton";

export const metadata: Metadata = {
  title: "ProTrader AI",
  description: "Indian stock predictions, backtests, and paper trading.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <header className="sticky top-0 z-10 border-b border-border bg-bg/80 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <Link href="/" className="font-semibold tracking-tight">
              ProTrader <span className="text-accent">AI</span>
            </Link>
            <nav className="flex items-center gap-4 text-sm">
              {user ? (
                <>
                  <Link href="/dashboard" className="text-muted hover:text-fg">
                    Dashboard
                  </Link>
                  <Link href="/accuracy" className="text-muted hover:text-fg">
                    Accuracy
                  </Link>
                  <span className="chip">{user.email}</span>
                  <SignOutButton />
                </>
              ) : (
                <Link href="/login" className="btn btn-primary">
                  Sign in
                </Link>
              )}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
