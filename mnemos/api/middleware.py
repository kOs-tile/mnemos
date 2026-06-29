"""
MNEMOS API Middleware.

Provides:
- AgentIdentityMiddleware: extracts agent_id from X-Agent-ID header,
  enforces per-profile memory isolation, validates API secret if configured.
- RequestLoggingMiddleware: structured request/response logging with timing.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from mnemos.config import get_settings


class AgentIdentityMiddleware(BaseHTTPMiddleware):
    """
    Extracts and validates agent identity from incoming requests.

    Headers:
        X-Agent-ID: Unique identifier for the calling agent.
                    Required for /ingest, /query, and /agent/* endpoints.
        X-Mnemos-Secret: Optional shared secret for service-to-service auth.
                         Only enforced if MNEMOS_API_SECRET is set in config.

    Injects into request.state:
        agent_id (str | None): Validated agent identifier.
        request_id (str): UUID for request tracing.
    """

    # Paths that don't require agent identity
    PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, **kwargs):
        super().__init__(app, **kwargs)
        self._settings = get_settings()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Assign a request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        path = request.url.path

        # Skip identity check for public paths
        if path in self.PUBLIC_PATHS or path.startswith("/docs"):
            request.state.agent_id = None
            return await call_next(request)

        # Check API secret if configured
        if self._settings.mnemos_api_secret:
            secret_header = request.headers.get("X-Mnemos-Secret", "")
            if secret_header != self._settings.mnemos_api_secret:
                logger.warning(
                    f"[{request_id}] Unauthorized request to {path}: invalid secret"
                )
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing X-Mnemos-Secret header"},
                )

        # Extract agent_id
        agent_id = request.headers.get("X-Agent-ID")

        # For routes that require an agent context, enforce the header
        requires_agent = any(
            path.startswith(prefix)
            for prefix in ["/ingest", "/query", "/agent/"]
        )

        if requires_agent and not agent_id:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "X-Agent-ID header is required for this endpoint",
                    "tip": "Set X-Agent-ID to your agent's unique identifier",
                },
            )

        # Sanitize: strip whitespace, truncate to 128 chars
        if agent_id:
            agent_id = agent_id.strip()[:128]

        request.state.agent_id = agent_id
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured request/response logging middleware.

    Logs:
    - Method, path, agent_id
    - Response status code
    - Processing time in ms

    Uses loguru for structured JSON-compatible output.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.perf_counter()
        agent_id = getattr(request.state, "agent_id", None)
        request_id = getattr(request.state, "request_id", "?")

        logger.debug(
            f"→ [{request_id}] {request.method} {request.url.path} "
            f"agent={agent_id}"
        )

        try:
            response = await call_next(request)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            log_fn = logger.debug if response.status_code < 400 else logger.warning
            log_fn(
                f"← [{request_id}] {response.status_code} "
                f"{request.method} {request.url.path} "
                f"{elapsed_ms}ms agent={agent_id}"
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time-Ms"] = str(elapsed_ms)
            return response

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                f"✗ [{request_id}] ERROR {request.method} {request.url.path} "
                f"{elapsed_ms}ms: {e}"
            )
            raise
