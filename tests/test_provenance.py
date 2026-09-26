from datetime import datetime, timedelta, timezone

from mnemos.memory.retrieval import _memory_allowed, _memory_rejection_reason
from mnemos.models import (
    MemoryNode,
    MemoryProvenance,
    MemoryStatus,
    MemoryType,
    QueryRequest,
    QueryResponse,
)


def make_node(
    *,
    provenance: MemoryProvenance = MemoryProvenance.UNKNOWN,
    valid_until=None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    source_trace_id: str | None = "trace-1",
) -> MemoryNode:
    return MemoryNode(
        id="00000000-0000-0000-0000-000000000001",
        type=MemoryType.SEMANTIC,
        content="A remembered claim",
        agent_id="agent-1",
        provenance=provenance,
        valid_until=valid_until,
        status=status,
        source_trace_id=source_trace_id,
    )


def test_expired_memory_is_stale():
    node = make_node(
        valid_until=datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    assert node.is_evidence_stale() is True


def test_future_validity_is_not_stale():
    node = make_node(
        valid_until=datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    assert node.is_evidence_stale() is False


def test_stale_memory_fails_closed_by_default():
    node = make_node(
        provenance=MemoryProvenance.EXTERNAL_EVIDENCE,
        valid_until=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    request = QueryRequest(query="claim", agent_id="agent-1")

    assert _memory_allowed(node, request) is False


def test_stale_memory_requires_explicit_opt_in():
    node = make_node(
        valid_until=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    request = QueryRequest(
        query="claim",
        agent_id="agent-1",
        include_stale_evidence=True,
    )

    assert _memory_allowed(node, request) is True


def test_provenance_allowlist_filters_derived_memory():
    node = make_node(provenance=MemoryProvenance.LLM_DERIVED)
    request = QueryRequest(
        query="claim",
        agent_id="agent-1",
        allowed_provenance=[MemoryProvenance.AGENT_TRACE],
    )

    assert _memory_allowed(node, request) is False


def test_provenance_allowlist_accepts_matching_memory():
    node = make_node(provenance=MemoryProvenance.AGENT_TRACE)
    request = QueryRequest(
        query="claim",
        agent_id="agent-1",
        allowed_provenance=[MemoryProvenance.AGENT_TRACE],
    )

    assert _memory_allowed(node, request) is True


def test_prompt_exposes_provenance_and_staleness():
    node = make_node(provenance=MemoryProvenance.LLM_DERIVED)
    response = QueryResponse(
        query="claim",
        memories=[node],
        retrieval_ms=1,
    )
    prompt = response.format_for_prompt()

    assert "provenance=llm_derived" in prompt
    assert "stale=false" in prompt
    assert "source_trace=trace-1" in prompt
    assert "not authorization" in prompt.lower()



def test_superseded_memory_fails_closed_defensively():
    node = make_node(status=MemoryStatus.SUPERSEDED)
    request = QueryRequest(query="claim", agent_id="agent-1")

    assert _memory_allowed(node, request) is False
    assert _memory_rejection_reason(node, request) == "inactive_status"


def test_archived_memory_fails_closed_defensively():
    node = make_node(status=MemoryStatus.ARCHIVED)
    request = QueryRequest(query="claim", agent_id="agent-1")

    assert _memory_allowed(node, request) is False
    assert _memory_rejection_reason(node, request) == "inactive_status"


def test_traceable_mode_rejects_memory_without_source_trace():
    node = make_node(
        provenance=MemoryProvenance.LLM_DERIVED,
        source_trace_id=None,
    )
    request = QueryRequest(
        query="claim",
        agent_id="agent-1",
        require_source_trace=True,
    )

    assert _memory_allowed(node, request) is False
    assert _memory_rejection_reason(node, request) == "missing_source_trace"


def test_traceable_mode_accepts_memory_with_source_trace():
    node = make_node(
        provenance=MemoryProvenance.LLM_DERIVED,
        source_trace_id="trace-grounded",
    )
    request = QueryRequest(
        query="claim",
        agent_id="agent-1",
        require_source_trace=True,
    )

    assert _memory_allowed(node, request) is True
