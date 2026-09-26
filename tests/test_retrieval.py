"""
Tests for QueryPlanner — hybrid vector + graph retrieval.

Uses mock Neo4j client and Qdrant client to test:
- Basic query pipeline
- Qdrant filtering by agent_id and memory type
- Graph traversal expansion
- Node ranking and deduplication
- Score composition (vector × salience × proximity)
- Access count updates
- Edge cases: empty results, Qdrant failure fallback
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

from mnemos.memory.retrieval import QueryPlanner, ScoredNode
from mnemos.models import (
    MemoryNode,
    MemoryType,
    QueryRequest,
    QueryResponse,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_node(
    node_id: str,
    content: str,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    agent_id: str = "test-agent",
    salience: float = 0.8,
) -> MemoryNode:
    return MemoryNode(
        id=node_id,
        type=memory_type,
        content=content,
        agent_id=agent_id,
        salience=salience,
        confidence=0.9,
        access_count=0,
    )


@pytest.fixture
def mock_nodes() -> list[MemoryNode]:
    return [
        _make_node("n001", "AWS EC2 t3.medium costs $0.0416/hour", salience=0.9),
        _make_node("n002", "Stripe was founded in 2010 by Patrick Collison", salience=0.7),
        _make_node("n003", "Python retry pattern with exponential backoff", MemoryType.PROCEDURAL, salience=0.85),
        _make_node("n004", "Competitor analysis: Notion vs Linear", MemoryType.EPISODIC, salience=0.6),
    ]


@pytest.fixture
def mock_neo4j(mock_nodes):
    client = AsyncMock()

    async def find_nodes_by_content_similarity(node_ids, agent_id, limit=50):
        return [n for n in mock_nodes if n.id in node_ids]

    async def traverse(start_node_ids, max_hops, agent_id=None, min_weight=0.0):
        # Return all nodes not in start_node_ids as "neighbors"
        return [n for n in mock_nodes if n.id not in start_node_ids]

    client.find_nodes_by_content_similarity = AsyncMock(
        side_effect=find_nodes_by_content_similarity
    )
    client.traverse = AsyncMock(side_effect=traverse)
    client.mark_accessed = AsyncMock()
    return client


@pytest.fixture
def mock_qdrant(mock_nodes):
    client = AsyncMock()

    def _make_search_result(node_id: str, score: float):
        r = MagicMock()
        r.payload = {"node_id": node_id, "agent_id": "test-agent", "type": "semantic", "status": "active"}
        r.score = score
        return r

    client.search = AsyncMock(
        return_value=[
            _make_search_result("n001", 0.95),
            _make_search_result("n002", 0.82),
        ]
    )
    return client


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=[0.1] * 384)
    return embedder


@pytest.fixture
def settings():
    from mnemos.config import Settings
    return Settings(
        qdrant_collection="test_collection",
        contradiction_similarity_threshold=0.85,
    )


@pytest.fixture
def query_planner(mock_neo4j, mock_qdrant, mock_embedder, settings):
    return QueryPlanner(
        neo4j_client=mock_neo4j,
        qdrant_client=mock_qdrant,
        embedder=mock_embedder,
        settings=settings,
    )


# ─── Basic Query Tests ────────────────────────────────────────────────────────

class TestQueryPlannerBasic:

    @pytest.mark.asyncio
    async def test_query_returns_query_response(self, query_planner):
        request = QueryRequest(query="AWS pricing", agent_id="test-agent")
        result = await query_planner.query(request)
        assert isinstance(result, QueryResponse)

    @pytest.mark.asyncio
    async def test_query_returns_memories(self, query_planner, mock_nodes):
        request = QueryRequest(query="AWS pricing", agent_id="test-agent", top_k=5)
        result = await query_planner.query(request)
        assert len(result.memories) > 0

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self, query_planner):
        request = QueryRequest(query="anything", agent_id="test-agent", top_k=1)
        result = await query_planner.query(request)
        assert len(result.memories) <= 1

    @pytest.mark.asyncio
    async def test_query_populates_formatted_context(self, query_planner):
        request = QueryRequest(query="AWS pricing", agent_id="test-agent")
        result = await query_planner.query(request)
        assert result.formatted_context != ""
        assert len(result.formatted_context) > 10

    @pytest.mark.asyncio
    async def test_query_includes_retrieval_ms(self, query_planner):
        request = QueryRequest(query="AWS pricing", agent_id="test-agent")
        result = await query_planner.query(request)
        assert result.retrieval_ms >= 0

    @pytest.mark.asyncio
    async def test_query_includes_graph_context(self, query_planner):
        request = QueryRequest(
            query="AWS pricing",
            agent_id="test-agent",
            include_graph_hops=2,
        )
        result = await query_planner.query(request)
        assert isinstance(result.graph_context, list)

    @pytest.mark.asyncio
    async def test_mark_accessed_called(self, query_planner, mock_neo4j):
        request = QueryRequest(query="AWS pricing", agent_id="test-agent")
        await query_planner.query(request)
        mock_neo4j.mark_accessed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedder_called_with_query(self, query_planner, mock_embedder):
        request = QueryRequest(query="AWS pricing", agent_id="test-agent")
        await query_planner.query(request)
        mock_embedder.encode.assert_called_once_with("AWS pricing", convert_to_list=True)


# ─── Qdrant Integration ───────────────────────────────────────────────────────

class TestQueryPlannerQdrant:

    @pytest.mark.asyncio
    async def test_qdrant_search_called_with_agent_filter(
        self, query_planner, mock_qdrant
    ):
        """Qdrant search must include agent_id in the filter."""
        request = QueryRequest(query="test query", agent_id="test-agent")
        await query_planner.query(request)

        mock_qdrant.search.assert_called()
        call_kwargs = mock_qdrant.search.call_args[1]
        query_filter = call_kwargs.get("query_filter")
        assert query_filter is not None

        # Check agent_id is in the filter
        filter_str = str(query_filter)
        assert "test-agent" in filter_str

    @pytest.mark.asyncio
    async def test_qdrant_failure_returns_empty_not_raises(
        self, mock_neo4j, mock_embedder, settings
    ):
        """If Qdrant fails, the query should return empty results gracefully."""
        failing_qdrant = AsyncMock()
        failing_qdrant.search = AsyncMock(side_effect=Exception("Qdrant unavailable"))

        planner = QueryPlanner(
            neo4j_client=mock_neo4j,
            qdrant_client=failing_qdrant,
            embedder=mock_embedder,
            settings=settings,
        )
        request = QueryRequest(query="test", agent_id="test-agent", include_graph_hops=0)
        result = await planner.query(request)
        # Should not raise, should return empty memories
        assert isinstance(result, QueryResponse)
        assert result.memories == []

    @pytest.mark.asyncio
    async def test_multiple_memory_types_use_or_semantics(
        self, mock_neo4j, mock_embedder, settings, mock_nodes
    ):
        """Multiple requested types must not degrade into an unfiltered search."""
        qdrant = AsyncMock()

        def result(node_id: str, score: float, type_value: str):
            row = MagicMock()
            row.payload = {
                "node_id": node_id,
                "agent_id": "test-agent",
                "type": type_value,
                "status": "active",
            }
            row.score = score
            return row

        qdrant.search = AsyncMock(
            side_effect=[
                [result("n003", 0.91, "procedural")],
                [result("n004", 0.87, "episodic")],
            ]
        )
        mock_neo4j.find_nodes_by_content_similarity = AsyncMock(
            return_value=[mock_nodes[2], mock_nodes[3]]
        )
        mock_neo4j.traverse = AsyncMock(return_value=[])

        planner = QueryPlanner(
            neo4j_client=mock_neo4j,
            qdrant_client=qdrant,
            embedder=mock_embedder,
            settings=settings,
        )
        request = QueryRequest(
            query="workflow history",
            agent_id="test-agent",
            memory_types=[MemoryType.PROCEDURAL, MemoryType.EPISODIC],
            include_graph_hops=0,
        )
        response = await planner.query(request)

        assert qdrant.search.await_count == 2
        assert {m.type for m in response.memories} == {
            MemoryType.PROCEDURAL,
            MemoryType.EPISODIC,
        }

    @pytest.mark.asyncio
    async def test_memory_type_filter_applied(self, query_planner, mock_qdrant):
        """When memory_types filter is set to a single type, Qdrant filter should include it."""
        request = QueryRequest(
            query="test",
            agent_id="test-agent",
            memory_types=[MemoryType.PROCEDURAL],
        )
        await query_planner.query(request)
        call_kwargs = mock_qdrant.search.call_args[1]
        query_filter = call_kwargs.get("query_filter")
        filter_str = str(query_filter)
        assert "procedural" in filter_str


# ─── Graph Traversal ─────────────────────────────────────────────────────────

class TestQueryPlannerGraphTraversal:

    @pytest.mark.asyncio
    async def test_graph_expansion_respects_memory_type_filter(
        self, mock_neo4j, mock_qdrant, mock_embedder, settings, mock_nodes
    ):
        mock_qdrant.search = AsyncMock(
            return_value=[
                MagicMock(
                    payload={
                        "node_id": "n003",
                        "agent_id": "test-agent",
                        "type": "procedural",
                        "status": "active",
                    },
                    score=0.95,
                )
            ]
        )
        mock_neo4j.find_nodes_by_content_similarity = AsyncMock(
            return_value=[mock_nodes[2]]
        )
        mock_neo4j.traverse = AsyncMock(
            return_value=[mock_nodes[0], mock_nodes[3]]
        )

        planner = QueryPlanner(
            neo4j_client=mock_neo4j,
            qdrant_client=mock_qdrant,
            embedder=mock_embedder,
            settings=settings,
        )
        response = await planner.query(
            QueryRequest(
                query="retry procedure",
                agent_id="test-agent",
                memory_types=[MemoryType.PROCEDURAL],
                include_graph_hops=1,
                top_k=10,
            )
        )

        assert [m.type for m in response.memories] == [MemoryType.PROCEDURAL]

    @pytest.mark.asyncio
    async def test_graph_hops_zero_skips_traverse(self, query_planner, mock_neo4j):
        request = QueryRequest(
            query="test",
            agent_id="test-agent",
            include_graph_hops=0,
        )
        await query_planner.query(request)
        mock_neo4j.traverse.assert_not_called()

    @pytest.mark.asyncio
    async def test_graph_hops_triggers_traverse(self, query_planner, mock_neo4j):
        request = QueryRequest(
            query="test",
            agent_id="test-agent",
            include_graph_hops=2,
        )
        await query_planner.query(request)
        mock_neo4j.traverse.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_neighbors_added_to_results(
        self, mock_neo4j, mock_qdrant, mock_embedder, settings, mock_nodes
    ):
        """Graph traversal results should be included in final memories."""
        # Qdrant returns only n001
        mock_qdrant.search = AsyncMock(
            return_value=[
                MagicMock(
                    payload={"node_id": "n001", "agent_id": "test-agent",
                              "type": "semantic", "status": "active"},
                    score=0.95,
                )
            ]
        )
        mock_neo4j.find_nodes_by_content_similarity = AsyncMock(
            return_value=[mock_nodes[0]]  # Only n001
        )
        # Traverse returns n002, n003
        mock_neo4j.traverse = AsyncMock(return_value=mock_nodes[1:3])

        planner = QueryPlanner(
            neo4j_client=mock_neo4j,
            qdrant_client=mock_qdrant,
            embedder=mock_embedder,
            settings=settings,
        )
        request = QueryRequest(
            query="test",
            agent_id="test-agent",
            top_k=10,
            include_graph_hops=1,
        )
        result = await planner.query(request)
        result_ids = {m.id for m in result.memories}
        # n001 from vector search + n002, n003 from graph
        assert "n001" in result_ids
        assert "n002" in result_ids or "n003" in result_ids


# ─── Ranking and Deduplication ────────────────────────────────────────────────

class TestQueryPlannerRanking:

    def test_scored_node_composite_score(self):
        """ScoredNode composite score should weight vector_score, salience, and distance."""
        node = _make_node("n001", "test", salience=1.0)
        sn = ScoredNode(node=node, vector_score=1.0, graph_distance=0)
        # Max score: 0.6 * 1.0 + 0.3 * 1.0 + 0.1 * 1.0 = 1.0
        assert abs(sn.composite_score - 1.0) < 1e-9

    def test_higher_vector_score_wins(self):
        """Higher vector score should yield higher composite score."""
        n = _make_node("n001", "test", salience=0.5)
        sn_low = ScoredNode(node=n, vector_score=0.3, graph_distance=0)
        sn_high = ScoredNode(node=n, vector_score=0.9, graph_distance=0)
        assert sn_high.composite_score > sn_low.composite_score

    def test_graph_distance_reduces_score(self):
        """Nodes reached via graph traversal (distance > 0) should score lower."""
        n = _make_node("n001", "test", salience=0.8)
        sn_direct = ScoredNode(node=n, vector_score=0.5, graph_distance=0)
        sn_hop1 = ScoredNode(node=n, vector_score=0.5, graph_distance=1)
        assert sn_direct.composite_score > sn_hop1.composite_score

    def test_zero_distance_proximity_is_one(self):
        n = _make_node("n001", "test", salience=0.5)
        sn = ScoredNode(node=n, vector_score=0.5, graph_distance=0)
        # proximity_boost = max(0, 1 - 0 * 0.3) = 1.0
        expected = 0.6 * 0.5 + 0.3 * 0.5 + 0.1 * 1.0
        assert abs(sn.composite_score - expected) < 1e-9

    @pytest.mark.asyncio
    async def test_no_duplicate_nodes_in_results(
        self, mock_neo4j, mock_qdrant, mock_embedder, settings, mock_nodes
    ):
        """The same node should not appear twice in results."""
        # Both Qdrant and traversal return n001
        mock_qdrant.search = AsyncMock(
            return_value=[
                MagicMock(
                    payload={"node_id": "n001", "agent_id": "test-agent",
                              "type": "semantic", "status": "active"},
                    score=0.9,
                )
            ]
        )
        mock_neo4j.find_nodes_by_content_similarity = AsyncMock(
            return_value=[mock_nodes[0]]
        )
        mock_neo4j.traverse = AsyncMock(
            return_value=[mock_nodes[0], mock_nodes[1]]  # n001 again + n002
        )

        planner = QueryPlanner(
            neo4j_client=mock_neo4j,
            qdrant_client=mock_qdrant,
            embedder=mock_embedder,
            settings=settings,
        )
        request = QueryRequest(
            query="test",
            agent_id="test-agent",
            top_k=10,
            include_graph_hops=1,
        )
        result = await planner.query(request)
        ids = [m.id for m in result.memories]
        assert len(ids) == len(set(ids)), "Duplicate nodes found in results"

    @pytest.mark.asyncio
    async def test_min_salience_filter(
        self, mock_neo4j, mock_qdrant, mock_embedder, settings
    ):
        """Nodes with salience below min_salience should be excluded."""
        low_salience_node = _make_node("n_low", "low salience fact", salience=0.1)
        high_salience_node = _make_node("n_high", "high salience fact", salience=0.9)

        mock_qdrant.search = AsyncMock(
            return_value=[
                MagicMock(
                    payload={"node_id": "n_low", "agent_id": "test-agent",
                              "type": "semantic", "status": "active"},
                    score=0.95,
                ),
                MagicMock(
                    payload={"node_id": "n_high", "agent_id": "test-agent",
                              "type": "semantic", "status": "active"},
                    score=0.8,
                ),
            ]
        )
        mock_neo4j.find_nodes_by_content_similarity = AsyncMock(
            return_value=[low_salience_node, high_salience_node]
        )
        mock_neo4j.traverse = AsyncMock(return_value=[])

        planner = QueryPlanner(
            neo4j_client=mock_neo4j,
            qdrant_client=mock_qdrant,
            embedder=mock_embedder,
            settings=settings,
        )
        request = QueryRequest(
            query="test",
            agent_id="test-agent",
            top_k=10,
            min_salience=0.5,
            include_graph_hops=0,
        )
        result = await planner.query(request)
        result_ids = {m.id for m in result.memories}
        assert "n_low" not in result_ids
        assert "n_high" in result_ids


# ─── format_for_prompt ────────────────────────────────────────────────────────

class TestQueryResponseFormatForPrompt:

    def test_empty_memories_returns_placeholder(self):
        from mnemos.models import QueryResponse
        response = QueryResponse(query="test", memories=[], retrieval_ms=5)
        assert "[No relevant memories" in response.format_for_prompt()

    def test_memories_grouped_by_type(self):
        from mnemos.models import QueryResponse
        nodes = [
            _make_node("n1", "Semantic fact A", MemoryType.SEMANTIC),
            _make_node("n2", "Episodic event B", MemoryType.EPISODIC),
        ]
        response = QueryResponse(query="test", memories=nodes, retrieval_ms=5)
        ctx = response.format_for_prompt()
        assert "SEMANTIC" in ctx.upper()
        assert "EPISODIC" in ctx.upper()

    def test_content_included_in_output(self):
        from mnemos.models import QueryResponse
        nodes = [_make_node("n1", "AWS costs $0.0416/hour")]
        response = QueryResponse(query="test", memories=nodes, retrieval_ms=5)
        ctx = response.format_for_prompt()
        assert "AWS costs" in ctx
