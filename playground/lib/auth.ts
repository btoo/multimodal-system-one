import "server-only";
import { cookies } from "next/headers";
import { validSession } from "./tokens";
export const COOKIE = "miso_owner";
export async function isOwner() {
  return validSession(
    (await cookies()).get(COOKIE)?.value,
    process.env.SESSION_SECRET,
  );
}
export { sameOrigin } from "./origin";
