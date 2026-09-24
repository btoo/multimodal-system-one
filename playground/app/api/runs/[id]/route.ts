import { getRun } from "workflow/api";
import { isOwner } from "@/lib/auth";
import type { RunResult } from "@/lib/contracts";
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!(await isOwner()))
    return Response.json({ error: "Unlock live runs first." }, { status: 401 });
  const { id } = await params;
  if (!/^wrun_[a-zA-Z0-9_-]+$/.test(id))
    return Response.json({ error: "Invalid run ID." }, { status: 400 });
  try {
    const run = getRun<RunResult>(id);
    if (!(await run.exists))
      return Response.json({ error: "Run not found." }, { status: 404 });
    const status = await run.status;
    if (status === "completed")
      return Response.json(
        { id, status, result: await run.returnValue },
        { headers: { "Cache-Control": "no-store" } },
      );
    if (status === "failed" || status === "cancelled")
      return Response.json(
        {
          id,
          status,
          error:
            "This run did not complete. Check the media format and question vocabulary, then start a new run.",
        },
        { headers: { "Cache-Control": "no-store" } },
      );
    return Response.json(
      { id, status },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return Response.json(
      { error: "Run status is temporarily unavailable." },
      { status: 503 },
    );
  }
}
