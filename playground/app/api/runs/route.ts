import { start } from "workflow/api";
import { decisionRun } from "@/workflows/decision";
import { isOwner, sameOrigin } from "@/lib/auth";
import { boundedJSON } from "@/lib/body";
import { runInputSchema, MAX_BODY } from "@/lib/contracts";
export const maxDuration = 60;
export async function POST(request: Request) {
  if (!(await isOwner()))
    return Response.json({ error: "Unlock live runs first." }, { status: 401 });
  if (!sameOrigin(request))
    return Response.json({ error: "Invalid request origin." }, { status: 403 });
  try {
    const input = runInputSchema.parse(await boundedJSON(request, MAX_BODY));
    const run = await start(decisionRun, [input]);
    return Response.json(
      { id: run.runId, status: "pending" },
      { status: 202, headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    const { ZodError } = await import("zod");
    if (error instanceof ZodError)
      return Response.json(
        {
          error: error.issues
            .map((i) => `${i.path.join(".")}: ${i.message}`)
            .slice(0, 3)
            .join("; "),
        },
        { status: 400 },
      );
    return Response.json(
      { error: "Could not start this run. Check the input and try again." },
      { status: 400 },
    );
  }
}
