import { cookies } from "next/headers";
import { COOKIE, isOwner, sameOrigin } from "@/lib/auth";
import { equalSecret, issueSession } from "@/lib/tokens";
import { boundedJSON } from "@/lib/body";
export async function GET() {
  return Response.json(
    { unlocked: await isOwner() },
    { headers: { "Cache-Control": "no-store" } },
  );
}
export async function POST(request: Request) {
  if (!sameOrigin(request))
    return Response.json({ error: "Invalid request origin." }, { status: 403 });
  try {
    const { key } = await boundedJSON(request, 2048);
    const expected = process.env.PLAYGROUND_ACCESS_KEY,
      secret = process.env.SESSION_SECRET;
    if (
      typeof key !== "string" ||
      !expected ||
      !secret ||
      !equalSecret(key, expected)
    )
      return Response.json(
        { error: "That access key is not valid." },
        { status: 401 },
      );
    (await cookies()).set(COOKIE, issueSession(secret), {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/",
      maxAge: 7 * 86400,
    });
    return Response.json({ unlocked: true });
  } catch {
    return Response.json(
      { error: "Could not unlock this session." },
      { status: 400 },
    );
  }
}
export async function DELETE(request: Request) {
  if (!sameOrigin(request)) return new Response(null, { status: 403 });
  (await cookies()).delete(COOKIE);
  return Response.json({ unlocked: false });
}
