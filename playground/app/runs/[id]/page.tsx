import Link from "next/link";
import { Shell } from "@/components/shell";
import { RunResults } from "@/components/results";
import { isOwner } from "@/lib/auth";
export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const [{ id }, unlocked] = await Promise.all([params, isOwner()]);
  return (
    <Shell active="runs">
      <main className="page-content saved-run">
        <Link className="back-link" href="/runs">
          ← Run history
        </Link>
        <section className="page-heading">
          <div>
            <div className="eyebrow">SAVED EXPERIMENT</div>
            <h1>Run results</h1>
            <p className="mono">{id}</p>
          </div>
          <Link href="/" className="secondary-button">
            New experiment
          </Link>
        </section>
        <RunResults runId={id} unlocked={unlocked} />
      </main>
    </Shell>
  );
}
