"""
MnemosClient — Python SDK for MNEMOS memory service.

Provides a clean async interface for Hermes agents to:
- Ingest execution traces
- Query semantic memory
- Retrieve specific entities
- Manage memory lifecycle

Designed for use inside Kavi Claw skill implementations.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mnemos.models import (
    AgentMemoryProfile,
    AgentTrace,
    EntityResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    MemoryType,
)


class MnemosClientError(Exception):
    """Base exception for MnemosClient errors."""


class MnemosConnectionError(MnemosClientError):
    """Raised when the MNEMOS service is unreachable."""


class MnemosAPIError(MnemosClientError):
    """Raised when the MNEMOS API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"MNEMOS API error {status_code}: {detail}")


class MnemosClient:
    """
    Async Python client for the MNEMOS memory service.

    Usage:
        client = MnemosClient(base_url="http://localhost:8000", agent_id="my-agent")
        await client.ingest_trace(trace)
        result = await client.query("What do we know about France?")

    Context manager (auto-closes session):
        async with MnemosClient(...) as client:
            result = await client.query("AWS cost estimates")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        agent_id: str = "default-agent",
        api_secret: str = "",
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self._api_secret = api_secret
        self._timeout = timeout
        self._http_client = http_client
        self._owns_client = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                headers=self._default_headers(),
            )
        return self._http_client

    def _default_headers(self) -> dict[str, str]:
        headers = {"X-Agent-ID": self.agent_id}
        if self._api_secret:
            headers["X-Mnemos-Secret"] = self._api_secret
        return headers

    async def __aenter__(self) -> MnemosClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client and self._owns_client:
            await self._http_client.aclose()
            self._http_client = None

    # ── Core Methods ─────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(MnemosConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def ingest_trace(
        self,
        trace: AgentTrace,
        force_extract: bool = False,
    ) -> IngestResponse:
        """
        Ingest an AgentTrace into the MNEMOS memory graph.

        Triggers LLM fact extraction, Neo4j graph update,
        Qdrant vector indexing, and contradiction detection.

        Args:
            trace: The agent execution trace to ingest.
            force_extract: Re-run extraction even if trace_id was already processed.

        Returns:
            IngestResponse with counts of created nodes, edges, and resolved contradictions.
        """
        payload = IngestRequest(trace=trace, force_extract=force_extract)
        response = await self._post("/ingest", payload.model_dump())
        return IngestResponse(**response)

    @retry(
        retry=retry_if_exception_type(MnemosConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def query(
        self,
        query: str,
        top_k: int = 10,
        include_graph_hops: int = 2,
        memory_types: list[MemoryType] | list[str] | None = None,
        min_salience: float = 0.0,
        agent_id_override: str | None = None,
    ) -> QueryResponse:
        """
        Query the memory graph with a natural language question.

        Args:
            query: Natural language query string.
            top_k: Maximum number of memories to return.
            include_graph_hops: Depth of graph traversal from vector-matched seeds.
            memory_types: Filter to specific memory types. None = all types.
            min_salience: Minimum salience threshold for returned nodes.
            agent_id_override: Query another agent's memory (if permitted).

        Returns:
            QueryResponse with ranked memories and formatted context string.
        """
        # Normalize memory_types
        normalized_types = None
        if memory_types:
            normalized_types = [
                mt.value if isinstance(mt, MemoryType) else mt
                for mt in memory_types
            ]

        request = QueryRequest(
            query=query,
            agent_id=agent_id_override or self.agent_id,
            top_k=top_k,
            include_graph_hops=include_graph_hops,
            memory_types=[MemoryType(t) for t in normalized_types] if normalized_types else None,
            min_salience=min_salience,
        )
        response = await self._post("/query", request.model_dump())
        return QueryResponse(**response)

    async def get_entity(self, entity_id: str) -> EntityResponse:
        """
        Retrieve a specific memory entity with its graph neighborhood.

        Args:
            entity_id: The MemoryNode.id to retrieve.

        Returns:
            EntityResponse with node, edges, and neighbor nodes.
        """
        response = await self._get(f"/entity/{entity_id}")
        return EntityResponse(**response)

    async def get_agent_profile(
        self, agent_id: str | None = None
    ) -> AgentMemoryProfile:
        """
        Get an agent's memory profile (stats + top nodes).

        Args:
            agent_id: Agent to query. Defaults to this client's agent_id.
        """
        target = agent_id or self.agent_id
        response = await self._post(f"/agent/{target}/memory", {})
        return AgentMemoryProfile(**response)

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory node.

        Args:
            memory_id: The MemoryNode.id to delete.

        Returns:
            True if successfully deleted.
        """
        response = await self._delete(f"/memory/{memory_id}")
        return response.get("status") == "ok"

    async def health_check(self) -> dict[str, Any]:
        """Check MNEMOS service health."""
        return await self._get("/health")

    # ── Convenience Methods ───────────────────────────────────────────────────

    async def remember(self, fact: str, confidence: float = 0.9) -> IngestResponse:
        """
        Shorthand: store a single semantic fact as an agent trace.

        Useful for quick testing or injecting ground-truth facts.

        Args:
            fact: The fact string to remember.
            confidence: Confidence level for this fact.

        Returns:
            IngestResponse.
        """
        trace = AgentTrace(
            agent_id=self.agent_id,
            inputs={"fact": fact},
            tool_calls=[],
            outputs={"stored_fact": fact},
            metadata={"confidence": confidence, "source": "direct_remember"},
        )
        return await self.ingest_trace(trace)

    async def context_for_prompt(
        self, query: str, top_k: int = 5
    ) -> str:
        """
        Retrieve memory and return a formatted string ready for LLM injection.

        Args:
            query: What to query.
            top_k: How many memories to include.

        Returns:
            Formatted context string for prompt injection.
        """
        result = await self.query(query, top_k=top_k)
        return result.formatted_context

    # ── HTTP Helpers ──────────────────────────────────────────────────────────

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        try:
            resp = await client.post(
                path,
                json=payload,
                headers=self._default_headers(),
            )
            self._raise_for_status(resp)
            return resp.json()
        except httpx.ConnectError as e:
            raise MnemosConnectionError(
                f"Cannot connect to MNEMOS at {self.base_url}: {e}"
            ) from e

    async def _get(self, path: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            resp = await client.get(path, headers=self._default_headers())
            self._raise_for_status(resp)
            return resp.json()
        except httpx.ConnectError as e:
            raise MnemosConnectionError(
                f"Cannot connect to MNEMOS at {self.base_url}: {e}"
            ) from e

    async def _delete(self, path: str) -> dict[str, Any]:
        client = self._get_client()
        try:
            resp = await client.delete(path, headers=self._default_headers())
            self._raise_for_status(resp)
            return resp.json()
        except httpx.ConnectError as e:
            raise MnemosConnectionError(
                f"Cannot connect to MNEMOS at {self.base_url}: {e}"
            ) from e

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise MnemosAPIError(response.status_code, detail)
