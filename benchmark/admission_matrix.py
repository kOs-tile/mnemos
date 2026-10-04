"""Expanded deterministic trust-boundary benchmark for MNEMOS."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mnemos.config import Settings
from mnemos.memory.retrieval import QueryPlanner, _memory_rejection_reason
from mnemos.models import (
    MemoryNode,
    MemoryProvenance,
    MemoryStatus,
    MemoryType,
    QueryRequest,
)


PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)
FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)


def _node(
    node_id: str,
    *,
    provenance: MemoryProvenance,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    valid_until=FUTURE,
    source_trace_id: str | None = "trace-1",
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryNode:
    return MemoryNode(
        id=node_id,
        type=memory_type,
        content=f"Benchmark memory {node_id}",
        agent_id="benchmark-agent",
        provenance=provenance,
        status=status,
        valid_until=valid_until,
        source_trace_id=source_trace_id,
        salience=0.9,
    )


def _request(
    *,
    include_stale_evidence: bool = False,
    allowed_provenance=None,
    require_source_trace: bool = False,
    include_graph_hops: int = 0,
) -> QueryRequest:
    return QueryRequest(
        query="benchmark memory policy",
        agent_id="benchmark-agent",
        top_k=10,
        include_stale_evidence=include_stale_evidence,
        allowed_provenance=allowed_provenance,
        require_source_trace=require_source_trace,
        include_graph_hops=include_graph_hops,
    )


def build_policy_cases():
    strict = _request(
        allowed_provenance=[
            MemoryProvenance.AGENT_TRACE,
            MemoryProvenance.EXTERNAL_EVIDENCE,
        ],
        require_source_trace=True,
    )
    allow_stale = _request(
        include_stale_evidence=True,
        allowed_provenance=[
            MemoryProvenance.AGENT_TRACE,
            MemoryProvenance.EXTERNAL_EVIDENCE,
        ],
        require_source_trace=True,
    )
    source_optional = _request(
        allowed_provenance=[
            MemoryProvenance.AGENT_TRACE,
            MemoryProvenance.EXTERNAL_EVIDENCE,
        ],
        require_source_trace=False,
    )
    no_provenance_policy = _request()
    only_trace = _request(
        allowed_provenance=[MemoryProvenance.AGENT_TRACE],
        require_source_trace=False,
    )

    return [
        ("fresh_agent_trace", _node("p01", provenance=MemoryProvenance.AGENT_TRACE), strict, None),
        ("fresh_external", _node("p02", provenance=MemoryProvenance.EXTERNAL_EVIDENCE), strict, None),
        ("fresh_llm_derived", _node("p03", provenance=MemoryProvenance.LLM_DERIVED), strict, "provenance_not_allowed"),
        ("fresh_user_asserted", _node("p04", provenance=MemoryProvenance.USER_ASSERTED), strict, "provenance_not_allowed"),
        ("fresh_unknown", _node("p05", provenance=MemoryProvenance.UNKNOWN), strict, "provenance_not_allowed"),
        ("expired_external", _node("p06", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, valid_until=PAST), strict, "stale_evidence"),
        ("expired_agent_trace", _node("p07", provenance=MemoryProvenance.AGENT_TRACE, valid_until=PAST), strict, "stale_evidence"),
        ("expired_external_opt_in", _node("p08", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, valid_until=PAST), allow_stale, None),
        ("expired_llm_opt_in", _node("p09", provenance=MemoryProvenance.LLM_DERIVED, valid_until=PAST), allow_stale, "provenance_not_allowed"),
        ("missing_trace_agent", _node("p10", provenance=MemoryProvenance.AGENT_TRACE, source_trace_id=None), strict, "missing_source_trace"),
        ("missing_trace_external", _node("p11", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, source_trace_id=None), strict, "missing_source_trace"),
        ("missing_trace_optional", _node("p12", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, source_trace_id=None), source_optional, None),
        ("archived", _node("p13", provenance=MemoryProvenance.AGENT_TRACE, status=MemoryStatus.ARCHIVED), strict, "inactive_status"),
        ("superseded", _node("p14", provenance=MemoryProvenance.AGENT_TRACE, status=MemoryStatus.SUPERSEDED), strict, "inactive_status"),
        ("inactive_precedence", _node("p15", provenance=MemoryProvenance.LLM_DERIVED, status=MemoryStatus.SUPERSEDED, valid_until=PAST), strict, "inactive_status"),
        ("stale_precedence", _node("p16", provenance=MemoryProvenance.LLM_DERIVED, valid_until=PAST), strict, "stale_evidence"),
        ("provenance_precedence", _node("p17", provenance=MemoryProvenance.LLM_DERIVED, source_trace_id=None), strict, "provenance_not_allowed"),
        ("unknown_without_policy", _node("p18", provenance=MemoryProvenance.UNKNOWN), no_provenance_policy, None),
        ("user_asserted_trace_only", _node("p19", provenance=MemoryProvenance.USER_ASSERTED), only_trace, "provenance_not_allowed"),
        ("agent_trace_trace_only", _node("p20", provenance=MemoryProvenance.AGENT_TRACE), only_trace, None),
    ]


class _Embedder:
    def encode(self, _query, *, convert_to_list):
        assert convert_to_list is True
        return [0.0] * 384


async def _graph_case(label: str, neighbor: MemoryNode, request: QueryRequest, expected_reason):
    seed = _node("seed", provenance=MemoryProvenance.AGENT_TRACE)
    qdrant = AsyncMock()
    qdrant.search = AsyncMock(
        return_value=[
            SimpleNamespace(
                payload={
                    "node_id": seed.id,
                    "agent_id": "benchmark-agent",
                    "type": "semantic",
                    "status": "active",
                },
                score=0.99,
            )
        ]
    )

    neo4j = AsyncMock()
    neo4j.find_nodes_by_content_similarity = AsyncMock(return_value=[seed])
    neo4j.traverse = AsyncMock(return_value=[neighbor])
    neo4j.mark_accessed = AsyncMock()

    planner = QueryPlanner(
        neo4j_client=neo4j,
        qdrant_client=qdrant,
        embedder=_Embedder(),
        settings=Settings(),
    )
    response = await planner.query(request)
    returned_ids = {node.id for node in response.memories}
    actual_reason = None
    for reason, ids in response.admission_report.rejected_memory_ids.items():
        if neighbor.id in ids:
            actual_reason = reason
            break

    included = neighbor.id in returned_ids
    expected_included = expected_reason is None
    correct = (
        included == expected_included
        and actual_reason == expected_reason
        and response.admission_report.policy_leak_count == 0
    )
    return {
        "label": label,
        "expected_reason": expected_reason,
        "actual_reason": actual_reason,
        "expected_included": expected_included,
        "actual_included": included,
        "policy_leak_count": response.admission_report.policy_leak_count,
        "correct": correct,
    }


async def run_benchmark_async():
    policy_rows = []
    policy_correct = 0
    leaks = 0
    rejection_counts: dict[str, int] = {}

    for label, node, request, expected_reason in build_policy_cases():
        actual_reason = _memory_rejection_reason(node, request)
        correct = actual_reason == expected_reason
        policy_correct += int(correct)
        if expected_reason is not None and actual_reason is None:
            leaks += 1
        if actual_reason is not None:
            rejection_counts[actual_reason] = rejection_counts.get(actual_reason, 0) + 1
        policy_rows.append(
            {
                "label": label,
                "expected_reason": expected_reason,
                "actual_reason": actual_reason,
                "correct": correct,
            }
        )

    graph_request = _request(
        allowed_provenance=[
            MemoryProvenance.AGENT_TRACE,
            MemoryProvenance.EXTERNAL_EVIDENCE,
        ],
        require_source_trace=True,
        include_graph_hops=1,
    )
    graph_rows = [
        await _graph_case(
            "graph_stale_neighbor",
            _node("g01", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, valid_until=PAST),
            graph_request,
            "stale_evidence",
        ),
        await _graph_case(
            "graph_disallowed_neighbor",
            _node("g02", provenance=MemoryProvenance.LLM_DERIVED),
            graph_request,
            "provenance_not_allowed",
        ),
        await _graph_case(
            "graph_missing_trace_neighbor",
            _node("g03", provenance=MemoryProvenance.EXTERNAL_EVIDENCE, source_trace_id=None),
            graph_request,
            "missing_source_trace",
        ),
        await _graph_case(
            "graph_allowed_neighbor",
            _node("g04", provenance=MemoryProvenance.EXTERNAL_EVIDENCE),
            graph_request,
            None,
        ),
    ]

    return {
        "policy_cases": len(policy_rows),
        "policy_correct": policy_correct,
        "policy_accuracy": policy_correct / len(policy_rows),
        "policy_silent_trust_leaks": leaks,
        "rejection_counts": rejection_counts,
        "graph_cases": len(graph_rows),
        "graph_correct": sum(row["correct"] for row in graph_rows),
        "graph_policy_leaks": sum(row["policy_leak_count"] for row in graph_rows),
        "total_contract_checks": len(policy_rows) + len(graph_rows),
        "policy_results": policy_rows,
        "graph_results": graph_rows,
    }


def run_benchmark():
    return asyncio.run(run_benchmark_async())


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
