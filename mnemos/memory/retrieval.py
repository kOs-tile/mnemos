"""
QueryPlanner — hybrid memory retrieval.

Pipeline:
1. Embed the natural language query (sentence transformer)
2. Qdrant vector similarity search → candidate node IDs
3. Fetch full MemoryNodes from Neo4j by ID
4. Graph traversal from candidate seeds → expand context
5. Rank and deduplicate by salience × vector_score
6. Update access_count / last_accessed on retrieved nodes
7. Return QueryResponse with formatted prompt context
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from mnemos.config import Settings, get_settings
from mnemos.graph.neo4j_client import Neo4jClient
from mnemos.models import (
    MemoryNode,
    MemoryStatus,
    MemoryType,
    QueryRequest,
    QueryResponse,
    RetrievalAdmissionReport,
)


def _provenance_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _memory_rejection_reason(
    node: MemoryNode,
    request: QueryRequest,
) -> str | None:
    """Return the deterministic admission failure reason, or None when admitted."""
    status = node.status.value if hasattr(node.status, "value") else str(node.status)
    if status != MemoryStatus.ACTIVE.value:
        return "inactive_status"
    if not request.include_stale_evidence and node.is_evidence_stale():
        return "stale_evidence"
    if request.allowed_provenance:
        allowed = {_provenance_value(item) for item in request.allowed_provenance}
        if _provenance_value(node.provenance) not in allowed:
            return "provenance_not_allowed"
    if request.require_source_trace and not node.source_trace_id:
        return "missing_source_trace"
    return None


def _memory_allowed(node: MemoryNode, request: QueryRequest) -> bool:
    """Apply the same auditable admission policy to direct and graph retrieval."""
    return _memory_rejection_reason(node, request) is None


@dataclass
class ScoredNode:
    """A MemoryNode with an associated relevance score."""

    node: MemoryNode
    vector_score: float = 0.0
    graph_distance: int = 0  # Hops from a seed node (0 = direct vector match)

    @property
    def composite_score(self) -> float:
        """
        Combined relevance score:
        - Vector similarity (0–1) weighted 60%
        - Salience (0–1) weighted 30%
        - Graph proximity boost (diminishing with distance) 10%
        """
        proximity_boost = max(0.0, 1.0 - (self.graph_distance * 0.3))
        return (
            0.6 * self.vector_score
            + 0.3 * self.node.salience
            + 0.1 * proximity_boost
        )


class QueryPlanner:
    """
    Hybrid memory retrieval engine.

    Combines dense vector search (Qdrant) with structured graph traversal (Neo4j)
    to return the most relevant memory context for a natural language query.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: AsyncQdrantClient,
        embedder: SentenceTransformer,
        settings: Settings | None = None,
    ) -> None:
        self._neo4j = neo4j_client
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._settings = settings or get_settings()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a memory query and return ranked context bundle.
        """
        start_ms = int(time.time() * 1000)
        logger.debug(
            f"QueryPlanner.query: agent={request.agent_id}, "
            f"top_k={request.top_k}, hops={request.include_graph_hops}"
        )

        # ── Step 1: Embed the query ───────────────────────────────────────────
        query_vector = self._embedder.encode(request.query, convert_to_list=True)

        # ── Step 2: Vector search via Qdrant ─────────────────────────────────
        vector_results = await self._qdrant_search(
            vector=query_vector,
            agent_id=request.agent_id,
            top_k=request.top_k * 3,  # Over-fetch; we'll re-rank
            memory_types=request.memory_types,
        )
        logger.debug(f"Qdrant returned {len(vector_results)} candidates")

        # ── Step 3: Fetch full nodes from Neo4j ───────────────────────────────
        seed_node_ids = [r["node_id"] for r in vector_results]
        seed_nodes = await self._neo4j.find_nodes_by_content_similarity(
            node_ids=seed_node_ids,
            agent_id=request.agent_id,
            limit=request.top_k * 2,
        )

        # Build score map from vector search
        vector_score_map: dict[str, float] = {
            r["node_id"]: r["score"] for r in vector_results
        }

        rejected_by_reason: dict[str, int] = {}
        rejected_memory_ids: dict[str, list[str]] = {}
        evaluated_ids: set[str] = set()

        def admit(node: MemoryNode) -> bool:
            if node.id in evaluated_ids:
                return _memory_allowed(node, request)
            evaluated_ids.add(node.id)
            reason = _memory_rejection_reason(node, request)
            if reason is None:
                return True
            rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
            rejected_memory_ids.setdefault(reason, []).append(node.id)
            return False

        scored_nodes: list[ScoredNode] = []
        for n in seed_nodes:
            if n.salience < request.min_salience:
                continue
            if admit(n):
                scored_nodes.append(
                    ScoredNode(
                        node=n,
                        vector_score=vector_score_map.get(n.id, 0.0),
                        graph_distance=0,
                    )
                )

        # ── Step 4: Graph traversal expansion ────────────────────────────────
        if request.include_graph_hops > 0 and seed_node_ids:
            neighbor_nodes = await self._neo4j.traverse(
                start_node_ids=seed_node_ids[: min(10, len(seed_node_ids))],
                max_hops=request.include_graph_hops,
                agent_id=request.agent_id,
                min_weight=0.1,
            )

            # Assign graph-expanded nodes a lower initial vector score
            existing_ids = {sn.node.id for sn in scored_nodes}
            for n in neighbor_nodes:
                type_allowed = (
                    request.memory_types is None or n.type in request.memory_types
                )
                if (
                    n.id not in existing_ids
                    and n.salience >= request.min_salience
                    and type_allowed
                    and admit(n)
                ):
                    scored_nodes.append(
                        ScoredNode(node=n, vector_score=0.3, graph_distance=1)
                    )
                    existing_ids.add(n.id)

        # ── Step 5: Re-rank and deduplicate ───────────────────────────────────
        scored_nodes.sort(key=lambda x: x.composite_score, reverse=True)
        final_nodes = [sn.node for sn in scored_nodes[: request.top_k]]

        # ── Step 6: Update access stats ───────────────────────────────────────
        if final_nodes:
            await self._neo4j.mark_accessed([n.id for n in final_nodes])

        # ── Step 7: Build graph context paths ────────────────────────────────
        graph_context = self._build_graph_context(scored_nodes[: request.top_k])

        end_ms = int(time.time() * 1000)
        policy_leak_count = sum(
            1 for node in final_nodes
            if _memory_rejection_reason(node, request) is not None
        )
        admission_report = RetrievalAdmissionReport(
            evaluated_candidates=len(evaluated_ids),
            eligible_candidates=len(scored_nodes),
            returned_memories=len(final_nodes),
            rejected_by_reason=rejected_by_reason,
            rejected_memory_ids=rejected_memory_ids,
            policy_leak_count=policy_leak_count,
        )
        response = QueryResponse(
            query=request.query,
            memories=final_nodes,
            graph_context=graph_context,
            retrieval_ms=end_ms - start_ms,
            admission_report=admission_report,
        )
        response.formatted_context = response.format_for_prompt()

        logger.info(
            f"QueryPlanner: retrieved {len(final_nodes)} nodes in {end_ms - start_ms}ms "
            f"(rejected={sum(rejected_by_reason.values())}, policy_leaks={policy_leak_count})"
        )
        return response

    async def _qdrant_search(
        self,
        vector: list[float],
        agent_id: str,
        top_k: int,
        memory_types: list[MemoryType] | None,
    ) -> list[dict[str, Any]]:
        """
        Perform filtered vector search in Qdrant.
        Returns list of {node_id, score} dicts.
        """
        must_conditions = [
            FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
            FieldCondition(key="status", match=MatchValue(value="active")),
        ]

        async def search_one(extra_conditions: list[FieldCondition]) -> list[dict[str, Any]]:
            qdrant_filter = Filter(must=[*must_conditions, *extra_conditions])
            results = await self._qdrant.search(
                collection_name=self._settings.qdrant_collection,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            return [
                {"node_id": r.payload["node_id"], "score": r.score}
                for r in results
                if r.payload and "node_id" in r.payload
            ]

        try:
            if memory_types and len(memory_types) > 1:
                # Search each requested type independently, then merge by node id.
                # This preserves OR semantics without accidentally returning other types.
                merged: dict[str, float] = {}
                for memory_type in memory_types:
                    rows = await search_one([
                        FieldCondition(
                            key="type",
                            match=MatchValue(value=memory_type.value),
                        )
                    ])
                    for row in rows:
                        merged[row["node_id"]] = max(
                            row["score"],
                            merged.get(row["node_id"], float("-inf")),
                        )
                return [
                    {"node_id": node_id, "score": score}
                    for node_id, score in sorted(
                        merged.items(), key=lambda item: item[1], reverse=True
                    )[:top_k]
                ]

            extra_conditions: list[FieldCondition] = []
            if memory_types:
                extra_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=memory_types[0].value),
                    )
                )
            return await search_one(extra_conditions)
        except Exception as e:
            logger.warning(f"Qdrant search failed: {e}. Returning empty results.")
            return []

    def _build_graph_context(
        self, scored_nodes: list[ScoredNode]
    ) -> list[dict[str, Any]]:
        """
        Build a simplified graph context representation for API response.
        Groups nodes by type with their scores.
        """
        context: list[dict[str, Any]] = []
        for sn in scored_nodes:
            context.append(
                {
                    "node_id": sn.node.id,
                    "type": sn.node.type,
                    "content_preview": sn.node.content[:150],
                    "composite_score": round(sn.composite_score, 4),
                    "vector_score": round(sn.vector_score, 4),
                    "salience": round(sn.node.salience, 4),
                    "provenance": _provenance_value(sn.node.provenance),
                    "stale": sn.node.is_evidence_stale(),
                    "graph_distance": sn.graph_distance,
                }
            )
        return context
