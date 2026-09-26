"""Deterministic policy-admission benchmark for MNEMOS memory retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from mnemos.memory.retrieval import _memory_rejection_reason
from mnemos.models import (
    MemoryNode,
    MemoryProvenance,
    MemoryStatus,
    MemoryType,
    QueryRequest,
)


def _node(
    node_id: str,
    *,
    provenance: MemoryProvenance,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    valid_until=None,
    source_trace_id: str | None = "trace-1",
) -> MemoryNode:
    return MemoryNode(
        id=node_id,
        type=MemoryType.SEMANTIC,
        content=f"Benchmark memory {node_id}",
        agent_id="benchmark-agent",
        provenance=provenance,
        status=status,
        valid_until=valid_until,
        source_trace_id=source_trace_id,
    )


def run_benchmark():
    now = datetime.now(timezone.utc)
    request = QueryRequest(
        query="benchmark",
        agent_id="benchmark-agent",
        allowed_provenance=[
            MemoryProvenance.AGENT_TRACE,
            MemoryProvenance.EXTERNAL_EVIDENCE,
        ],
        require_source_trace=True,
    )

    cases = [
        (
            "fresh_trace",
            _node("fresh", provenance=MemoryProvenance.AGENT_TRACE),
            None,
        ),
        (
            "expired_external",
            _node(
                "expired",
                provenance=MemoryProvenance.EXTERNAL_EVIDENCE,
                valid_until=now - timedelta(seconds=1),
            ),
            "stale_evidence",
        ),
        (
            "disallowed_llm_derived",
            _node("derived", provenance=MemoryProvenance.LLM_DERIVED),
            "provenance_not_allowed",
        ),
        (
            "superseded",
            _node(
                "superseded",
                provenance=MemoryProvenance.AGENT_TRACE,
                status=MemoryStatus.SUPERSEDED,
            ),
            "inactive_status",
        ),
        (
            "archived",
            _node(
                "archived",
                provenance=MemoryProvenance.AGENT_TRACE,
                status=MemoryStatus.ARCHIVED,
            ),
            "inactive_status",
        ),
        (
            "trace_missing",
            _node(
                "trace-missing",
                provenance=MemoryProvenance.AGENT_TRACE,
                source_trace_id=None,
            ),
            "missing_source_trace",
        ),
    ]

    rows = []
    correct = 0
    leaks = 0
    for label, node, expected_reason in cases:
        actual_reason = _memory_rejection_reason(node, request)
        passed = actual_reason == expected_reason
        correct += int(passed)
        if expected_reason is not None and actual_reason is None:
            leaks += 1
        rows.append({
            "label": label,
            "expected_reason": expected_reason,
            "actual_reason": actual_reason,
            "correct": passed,
        })

    return {
        "cases": len(cases),
        "correct": correct,
        "policy_accuracy": correct / len(cases),
        "silent_trust_leaks": leaks,
        "silent_trust_rate": leaks / max(1, sum(1 for _, _, reason in cases if reason is not None)),
        "results": rows,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
