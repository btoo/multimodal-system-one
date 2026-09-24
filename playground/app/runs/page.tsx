import { Shell } from "@/components/shell";
import { RunHistory } from "@/components/history";
export default function Page() {
  return (
    <Shell active="runs">
      <main className="page-content">
        <section className="page-heading">
          <div>
            <div className="eyebrow">YOUR EXPERIMENTS</div>
            <h1>Run history</h1>
            <p>
              Recent runs from this browser. Reopen a result without calling the
              model again.
            </p>
          </div>
        </section>
        <RunHistory />
      </main>
    </Shell>
  );
}
