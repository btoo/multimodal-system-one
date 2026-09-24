import { Shell } from "@/components/shell";
import { Playground } from "@/components/playground";
import { isOwner } from "@/lib/auth";
export default async function Page({
  searchParams,
}: {
  searchParams: Promise<{ run?: string }>;
}) {
  const [query, unlocked] = await Promise.all([searchParams, isOwner()]);
  return (
    <Shell>
      <main className="page-content">
        <Playground unlocked={unlocked} initialRunId={query.run ?? null} />
      </main>
    </Shell>
  );
}
