import { createHmac, timingSafeEqual, randomBytes } from "node:crypto";
export function equalSecret(left: string, right: string) {
  const a = Buffer.from(left),
    b = Buffer.from(right);
  return a.length === b.length && timingSafeEqual(a, b);
}
export function issueSession(secret: string, now = Date.now()) {
  const data = Buffer.from(
    JSON.stringify({
      expires: now + 7 * 86400000,
      nonce: randomBytes(18).toString("hex"),
    }),
  ).toString("base64url");
  return `${data}.${createHmac("sha256", secret).update(data).digest("base64url")}`;
}
export function validSession(
  token: string | undefined,
  secret: string | undefined,
  now = Date.now(),
) {
  if (!token || !secret || token.length > 1024) return false;
  const [data, signature, ...rest] = token.split(".");
  if (!data || !signature || rest.length) return false;
  const expected = createHmac("sha256", secret)
    .update(data)
    .digest("base64url");
  if (!equalSecret(signature, expected)) return false;
  try {
    const value = JSON.parse(Buffer.from(data, "base64url").toString());
    return (
      typeof value.expires === "number" &&
      value.expires > now &&
      value.expires <= now + 8 * 86400000
    );
  } catch {
    return false;
  }
}
