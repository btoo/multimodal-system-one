"""FastAPI application; model weights are loaded once at process startup."""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
import threading
from types import MappingProxyType

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from .runtime import NativeRuntime
from .registry import registry_snapshot
from .schema import MODEL_ID, DecisionRequest, DecisionResponse

MAX_REQUEST_BYTES = 4 * 1024 * 1024


class BodyLimitMiddleware:
    """Bound a whole request before JSON parsing, including chunked transfers."""
    def __init__(self, app, limit=MAX_REQUEST_BYTES):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            data = message.get("body", b"")
            total += len(data)
            if total > self.limit:
                response = JSONResponse(status_code=413, content={"error": {
                    "code": "request_too_large", "message": f"Request body exceeds {self.limit} bytes"}})
                return await response(scope, receive, send)
            chunks.append(data)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return await self.app(scope, bounded_receive, send)


def create_app(device="auto", api_key=None, *, registry=None):
    key = os.environ.get("MMSO_API_KEY", "") if api_key is None else api_key
    registrations = registry_snapshot(registry)

    @asynccontextmanager
    async def lifespan(app):
        inference_lock = threading.Lock()
        runtimes = {name: NativeRuntime(device, registration, inference_lock=inference_lock)
                    for name, registration in registrations.items()}
        app.state.runtimes = MappingProxyType(runtimes)
        app.state.runtime = runtimes[MODEL_ID]  # Historical default-runtime Python attribute.
        try:
            yield
        finally:
            del app.state.runtime
            del app.state.runtimes

    app = FastAPI(title="MiSO Decisions", version="0.1.0",
        description="Experimental local audio/image/question classification. This is not an OpenAI, Claude, or Jev compatible server.",
        lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if key and (credentials is None or not hmac.compare_digest(credentials.credentials.encode("utf-8"), key.encode("utf-8"))):
            raise HTTPException(status_code=401, detail="A valid bearer API key is required", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Do not echo media payloads from Pydantic's error.input field.
        details = [{"path": ".".join(map(str, item["loc"])), "message": item["msg"], "type": item["type"]}
                   for item in error.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request",
            "message": "Request does not match the API schema", "details": details}})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, error):
        return JSONResponse(status_code=error.status_code, headers=error.headers,
            content={"error": {"code": {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(error.status_code, "request_error"),
                               "message": error.detail}})

    @app.get("/healthz")
    def health(request: Request):
        return {"status": "ready", "loaded_models": len(getattr(request.app.state, "runtimes", {}))}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models(request: Request):
        return {"object": "list", "data": [runtime.model_card() for runtime in request.app.state.runtimes.values()]}

    @app.get("/v1/models/{model_id}", dependencies=[Depends(authorize)])
    def model(model_id: str, request: Request):
        runtime = request.app.state.runtimes.get(model_id)
        if runtime is None:
            raise HTTPException(404, "Unknown model; see /v1/models")
        return runtime.model_card()

    @app.post("/v1/decisions", response_model=DecisionResponse, dependencies=[Depends(authorize)])
    def decide(body: DecisionRequest, request: Request):
        runtime = request.app.state.runtimes.get(body.model)
        if runtime is None:
            raise HTTPException(404, "Unknown model; see /v1/models")
        try:
            return runtime.decide(body)
        except ValueError as error:
            return JSONResponse(status_code=422, content={"error": {"code": "invalid_input", "message": str(error)}})

    return app
