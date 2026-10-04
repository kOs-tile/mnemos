"""Executable hybrid-retrieval admission corpus for MNEMOS.

The corpus exercises the real QueryPlanner path with deterministic in-memory
Qdrant/Neo4j stand-ins. It validates that direct vector candidates and
graph-expanded neighbors share the same admission policy.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from mnemos.config import Settings
from mnemos.memory.retrieval import QueryPlanner
from mnemos.models import (
    MemoryNode,
    MemoryProvenance,
    MemoryStatus,
    MemoryType,
    QueryRequest,
)


PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)
FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)
AGENT = "benchmark-agent"


def _node(
    node_id: str,
    *,
    provenance: MemoryProvenance,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    valid_until=None,
    source_trace_id: str | None = "trace-1",
    salience: float = 0.9,
) -> MemoryNode:
    return MemoryNode(
        id=node_id,
        type=MemoryType.SEMANTIC,
        content=f"Benchmark memory {node_id}",
        agent_id=AGENT,
        salience=salience,
        confidence=0.9,
        status=status,
        provenance=provenance,
        valid_until=valid_until,
        source_trace_id=source_trace_id,
    )


def build_nodes():
    direct = [
        _node("direct-safe-trace", provenance=MemoryProvenance.AGENT_TRACE),
        _node("direct-stale", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, valid_until=PAST),
        _node("direct-disallowed", provenance=MemoryProvenance.LLM_DERIVED),
        _node("direct-missing-trace", provenance=MemoryProvenance.AGENT_TRACE, source_trace_id=None),
    ]
    graph = [
        _node("graph-safe-trace", provenance=MemoryProvenance.AGENT_TRACE),
        _node("graph-safe-external", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, valid_until=FUTURE),
        _node("graph-stale", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, valid_until=PAST),
        _node("graph-disallowed", provenance=MemoryProvenance.USER_ASSERTED),
        _node("graph-missing-trace", provenance=MemoryProvenance.AGENT_TRACE, source_trace_id=None),
        _node(
            "graph-superseded",
            provenance=MemoryProvenance.AGENT_TRACE,
            status=MemoryStatus.SUPERSEDED,
        ),
    ]
    return direct, graph


def _planner():
    direct, graph = build_nodes()

    qdrant = AsyncMock()
    qdrant.search = AsyncMock(
        return_value=[
            SimpleNamespace(payload={"node_id": node.id}, score=0.99 - i * 0.03)
            for i, node in enumerate(direct)
        ]
    )

    neo4j = AsyncMock()

    async def find_nodes_by_content_similarity(node_ids, agent_id, limit=50):
        return [node for node in direct if node.id in node_ids]

    neo4j.find_nodes_by_content_similarity = AsyncMock(
        side_effect=find_nodes_by_content_similarity
    )
    neo4j.traverse = AsyncMock(return_value=graph)
    neo4j.mark_accessed = AsyncMock()

    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=[0.1] * 384)

    settings = Settings(
        qdrant_collection="benchmark_collection",
        contradiction_similarity_threshold=0.85,
    )
    return QueryPlanner(
        neo4j_client=neo4j,
        qdrant_client=qdrant,
        embedder=embedder,
        settings=settings,
    )


async def _run_scenario(label: str, request: QueryRequest) -> dict:
    planner = _planner()
    response = await planner.query(request)
    return {
        "label": label,
        "returned_ids": [memory.id for memory in response.memories],
        "returned_count": len(response.memories),
        "policy_leak_count": response.admission_report.policy_leak_count,
        "evaluated_candidates": response.admission_report.evaluated_candidates,
        "eligible_candidates": response.admission_report.eligible_candidates,
        "rejected_by_reason": response.admission_report.rejected_by_reason,
        "admission_fingerprint": response.admission_report.admission_fingerprint,
    }


async def run_benchmark_async() -> dict:
    strict = await _run_scenario(
        "strict_evidence_sensitive",
        QueryRequest(
            query="trusted evidence",
            agent_id=AGENT,
            top_k=20,
            include_graph_hops=1,
            include_stale_evidence=False,
            allowed_provenance=[
                MemoryProvenance.AGENT_TRACE,
                MemoryProvenance.EXTERNAL_EVIDENCE,
            ],
            require_source_trace=True,
        ),
    )

    stale_opt_in = await _run_scenario(
        "explicit_stale_opt_in",
        QueryRequest(
            query="trusted evidence including stale",
            agent_id=AGENT,
            top_k=20,
            include_graph_hops=1,
            include_stale_evidence=True,
            allowed_provenance=[
                MemoryProvenance.AGENT_TRACE,
                MemoryProvenance.EXTERNAL_EVIDENCE,
            ],
            require_source_trace=True,
        ),
    )

    advisory = await _run_scenario(
        "default_advisory",
        QueryRequest(
            query="memory context",
            agent_id=AGENT,
            top_k=20,
            include_graph_hops=1,
            include_stale_evidence=False,
            allowed_provenance=None,
            require_source_trace=False,
        ),
    )

    scenarios = [strict, stale_opt_in, advisory]
    return {
        "scenarios": len(scenarios),
        "policy_leaks": sum(row["policy_leak_count"] for row in scenarios),
        "results": scenarios,
    }


def run_benchmark() -> dict:
    return asyncio.run(run_benchmark_async())


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
