import Link from "next/link";
import { createClient } from "@/utils/supabase/server";
import { redirect } from "next/navigation";

export default async function Home() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (user) redirect("/dashboard");

  return (
    <div className="mx-auto max-w-2xl space-y-6 pt-12 text-center">
      <h1 className="text-4xl font-semibold tracking-tight">
        Predict Indian stocks with an AI ensemble.
      </h1>
      <p className="text-muted">
        FinBERT sentiment, option-chain features, macro overlays, and conformal
        intervals — served by a FastAPI backend with Celery workers and a
        Supabase auth layer.
      </p>
      <div className="flex items-center justify-center gap-3">
        <Link href="/login" className="btn btn-primary">
          Sign in
        </Link>
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="btn"
        >
          API docs
        </a>
      </div>
    </div>
  );
}
