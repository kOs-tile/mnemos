"""
HermesHook — Drop-in trace wrapper for Hermes skill execution.

Wraps Hermes skill calls to automatically capture:
- Inputs / outputs
- Tool call records
- Execution timing
- Errors and exceptions

And ingests them into MNEMOS after each skill execution.

Usage:
    hook = HermesHook(mnemos_url="http://localhost:8000", agent_id="kavi-orchestrator")

    @hook.trace_skill("web_search")
    async def web_search(query: str) -> dict:
        return await do_search(query)

    # Or decorate a full Hermes skill class:
    class MySkill(HermesHook.TracedSkillMixin):
        skill_name = "my_skill"
        ...
"""

from __future__ import annotations

import asyncio
import functools
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeVar

from loguru import logger

from mnemos.models import AgentTrace, ToolCall
from mnemos.sdk.client import MnemosClient

F = TypeVar("F", bound=Callable[..., Any])


class HermesHook:
    """
    Automatic memory ingestion hook for Hermes skill executions.

    Can be used as a:
    - Function decorator (@hook.trace_skill("skill_name"))
    - Context manager (async with hook.session(...))
    - Mixin for Hermes skill classes
    """

    def __init__(
        self,
        mnemos_url: str = "http://localhost:8000",
        agent_id: str = "hermes-agent",
        api_secret: str = "",
        client: MnemosClient | None = None,
        ingest_errors: bool = True,
        background_ingest: bool = True,
    ) -> None:
        """
        Args:
            mnemos_url: MNEMOS service URL.
            agent_id: Agent identifier for memory scoping.
            api_secret: MNEMOS API secret for authentication.
            client: Existing MnemosClient (avoids creating a new one).
            ingest_errors: Whether to ingest traces that contain errors.
            background_ingest: Run ingestion as a background task (non-blocking).
        """
        self.agent_id = agent_id
        self.ingest_errors = ingest_errors
        self.background_ingest = background_ingest

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = MnemosClient(
                base_url=mnemos_url,
                agent_id=agent_id,
                api_secret=api_secret,
            )
            self._owns_client = True

        self._active_session_id: str | None = None

    # ── Decorator API ─────────────────────────────────────────────────────────

    def trace_skill(
        self,
        skill_name: str,
        session_id: str | None = None,
    ) -> Callable[[F], F]:
        """
        Decorator: wraps an async skill function to auto-ingest traces.

        Args:
            skill_name: Name of the skill (used as tool name in trace).
            session_id: Optional fixed session ID. If None, inherits active session.

        Usage:
            @hook.trace_skill("web_search")
            async def web_search(query: str) -> dict:
                ...
        """

        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self._execute_traced(
                    fn=fn,
                    skill_name=skill_name,
                    args=args,
                    kwargs=kwargs,
                    session_id=session_id or self._active_session_id,
                )

            return wrapper  # type: ignore[return-value]

        return decorator

    # ── Context Manager API ───────────────────────────────────────────────────

    @asynccontextmanager
    async def session(
        self, session_id: str | None = None
    ) -> AsyncGenerator[HermesHook, None]:
        """
        Context manager: sets a shared session_id for all traced calls within scope.

        Usage:
            async with hook.session(session_id="sess_abc") as h:
                await some_traced_skill(...)
        """
        prev_session = self._active_session_id
        self._active_session_id = session_id or str(uuid.uuid4())
        try:
            yield self
        finally:
            self._active_session_id = prev_session

    # ── Direct Trace Ingestion ────────────────────────────────────────────────

    async def ingest(self, trace: AgentTrace) -> None:
        """
        Directly ingest a pre-built AgentTrace.

        Args:
            trace: The trace to ingest.
        """
        if self.background_ingest:
            asyncio.create_task(self._safe_ingest(trace))
        else:
            await self._safe_ingest(trace)

    async def _safe_ingest(self, trace: AgentTrace) -> None:
        """Ingest with error suppression (hook should never crash the agent)."""
        try:
            result = await self._client.ingest_trace(trace)
            logger.debug(
                f"HermesHook ingested trace {trace.trace_id}: "
                f"{result.nodes_created} nodes, {result.edges_created} edges"
            )
        except Exception as e:
            logger.warning(f"HermesHook ingestion failed (non-fatal): {e}")

    # ── Internal Execution Wrapper ────────────────────────────────────────────

    async def _execute_traced(
        self,
        fn: Callable,
        skill_name: str,
        args: tuple,
        kwargs: dict[str, Any],
        session_id: str | None = None,
    ) -> Any:
        """
        Execute fn(*args, **kwargs) and capture a full AgentTrace.
        """
        trace_id = str(uuid.uuid4())
        active_session = session_id or str(uuid.uuid4())
        start_ms = int(time.time() * 1000)

        # Build inputs dict from positional + keyword args
        inputs: dict[str, Any] = {}
        try:
            import inspect
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = dict(bound.arguments)
        except Exception:
            inputs = {"args": str(args)[:500], "kwargs": str(kwargs)[:500]}

        tool_call = ToolCall(
            tool=skill_name,
            args=inputs,
        )
        errors: list[str] = []
        outputs: dict[str, Any] = {}
        result: Any = None

        try:
            result = await fn(*args, **kwargs)
            tool_call.result = _safe_serialize(result)
            tool_call.duration_ms = int(time.time() * 1000) - start_ms
            outputs = {"result": _safe_serialize(result)}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            errors.append(error_msg)
            tool_call.error = error_msg
            tool_call.duration_ms = int(time.time() * 1000) - start_ms
            logger.debug(f"HermesHook: skill '{skill_name}' raised {error_msg}")

            if not self.ingest_errors:
                raise  # Don't ingest error traces if configured that way

            # Still ingest error traces (they contain valuable learning signal)
            trace = AgentTrace(
                trace_id=trace_id,
                agent_id=self.agent_id,
                session_id=active_session,
                inputs=inputs,
                tool_calls=[tool_call],
                outputs={},
                errors=errors,
                duration_ms=int(time.time() * 1000) - start_ms,
            )
            await self.ingest(trace)
            raise  # Re-raise after ingestion

        duration_ms = int(time.time() * 1000) - start_ms

        trace = AgentTrace(
            trace_id=trace_id,
            agent_id=self.agent_id,
            session_id=active_session,
            inputs=inputs,
            tool_calls=[tool_call],
            outputs=outputs,
            errors=errors,
            duration_ms=duration_ms,
        )
        await self.ingest(trace)
        return result

    # ── Mixin for Hermes Skill Classes ────────────────────────────────────────

    class TracedSkillMixin:
        """
        Mixin for Hermes Skill classes to auto-wrap execute() with MNEMOS tracing.

        Usage:
            class WebSearchSkill(HermesHook.TracedSkillMixin):
                skill_name = "web_search"
                mnemos_url = "http://localhost:8000"
                agent_id = "kavi-orchestrator"

                async def _execute(self, inputs: dict) -> dict:
                    # Your skill logic here
                    ...
        """

        skill_name: str = "unnamed_skill"
        mnemos_url: str = "http://localhost:8000"
        agent_id: str = "hermes-agent"
        _hook: HermesHook | None = None

        def get_hook(self) -> HermesHook:
            if self._hook is None:
                self._hook = HermesHook(
                    mnemos_url=self.mnemos_url,
                    agent_id=self.agent_id,
                )
            return self._hook

        async def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
            """Auto-traced execute wrapper."""
            hook = self.get_hook()
            return await hook._execute_traced(
                fn=self._execute,
                skill_name=self.skill_name,
                args=(inputs,),
                kwargs={},
                session_id=inputs.get("session_id"),
            )

        async def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
            """Override this in your skill implementation."""
            raise NotImplementedError

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying MnemosClient (if owned by this hook)."""
        if self._owns_client:
            await self._client.close()


def _safe_serialize(value: Any) -> Any:
    """Convert any value to a JSON-safe representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in list(value)[:50]]
    try:
        import json
        json.dumps(value)  # Test serializability
        return value
    except (TypeError, ValueError):
        return str(value)[:500]
