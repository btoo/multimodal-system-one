export function sameOrigin(request: Request) {
  const origin=request.headers.get("origin");
  if(!origin)return false;
  try {
    const observed=new URL(origin),target=new URL(request.url);
    return observed.host===(request.headers.get("host")??target.host)&&observed.protocol===target.protocol;
  } catch { return false; }
}
