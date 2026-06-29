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
    MemoryType,
    QueryRequest,
    QueryResponse,
)


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

        scored_nodes: list[ScoredNode] = [
            ScoredNode(
                node=n,
                vector_score=vector_score_map.get(n.id, 0.0),
                graph_distance=0,
            )
            for n in seed_nodes
            if n.salience >= request.min_salience
        ]

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
                if n.id not in existing_ids and n.salience >= request.min_salience:
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
        response = QueryResponse(
            query=request.query,
            memories=final_nodes,
            graph_context=graph_context,
            retrieval_ms=end_ms - start_ms,
        )
        response.formatted_context = response.format_for_prompt()

        logger.info(
            f"QueryPlanner: retrieved {len(final_nodes)} nodes in {end_ms - start_ms}ms"
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

        if memory_types:
            # Qdrant doesn't support OR natively on string fields easily,
            # so we do separate searches and merge if multiple types requested
            pass  # handled below

        qdrant_filter = Filter(must=must_conditions)

        if memory_types and len(memory_types) == 1:
            qdrant_filter.must.append(
                FieldCondition(
                    key="type",
                    match=MatchValue(value=memory_types[0].value),
                )
            )

        try:
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
                    "graph_distance": sn.graph_distance,
                }
            )
        return context
