"""
ContradictionResolver — detects conflicting facts and adjudicates via LLM micro-agent.

Pipeline:
1. Embedding-based candidate pruning — find pairs of semantic nodes with high cosine similarity
2. LLM adjudication — micro-agent receives both facts + provenance, decides winner
3. Graph update — winner confidence increases, loser gets SUPERSEDED_BY edge + status update

Design decision: We only run contradiction detection on newly ingested SEMANTIC nodes,
checking against the existing graph. This is cheaper than pairwise comparisons across
the full graph on every sweep.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mnemos.config import Settings, get_settings
from mnemos.graph.neo4j_client import Neo4jClient
from mnemos.models import (
    ContradictionReport,
    MemoryEdge,
    MemoryNode,
    MemoryStatus,
    RelationType,
)


_ADJUDICATION_SYSTEM_PROMPT = """You are a fact adjudication specialist for an AI memory system.
Two facts about the same topic may be in conflict. Your job is to:
1. Determine if they actually contradict each other (or are complementary/compatible)
2. If they contradict, decide which is more likely to be correct
3. Provide reasoning

Respond ONLY with JSON in this exact format:
{
    "is_contradiction": true | false,
    "winner_id": "<id of the more accurate fact, or null if not a contradiction>",
    "loser_id": "<id of the less accurate fact, or null if not a contradiction>",
    "reasoning": "<brief explanation>",
    "confidence": 0.0-1.0
}
"""


class ContradictionResolver:
    """
    Detects and resolves contradictory facts in the MNEMOS memory graph.

    Uses two-stage approach:
    1. Fast: Qdrant vector similarity to find candidate conflict pairs
    2. Slow: LLM micro-agent for semantic adjudication

    Designed to run after each ingestion batch on newly created semantic nodes.
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
        self._llm_client: AsyncOpenAI | None = None

    def _get_llm(self) -> AsyncOpenAI:
        if self._llm_client is None:
            self._llm_client = AsyncOpenAI(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
            )
        return self._llm_client

    async def detect_and_resolve(
        self,
        new_node_ids: list[str],
        agent_id: str,
        max_pairs: int = 10,
    ) -> ContradictionReport:
        """
        Main entry point: detect contradictions among new nodes vs existing graph.

        Args:
            new_node_ids: IDs of newly ingested semantic nodes.
            agent_id: Agent scope to restrict comparison to.
            max_pairs: Maximum number of conflict pairs to adjudicate per run.

        Returns:
            ContradictionReport with counts and resolution details.
        """
        if not new_node_ids:
            return ContradictionReport(candidate_pairs=0, resolved=0)

        # ── Step 1: Fetch newly created nodes ────────────────────────────────
        new_nodes: list[MemoryNode] = []
        for nid in new_node_ids:
            node = await self._neo4j.get_node(nid)
            if node:
                new_nodes.append(node)

        if not new_nodes:
            return ContradictionReport(candidate_pairs=0, resolved=0)

        # ── Step 2: Find candidate pairs via embedding similarity ─────────────
        candidate_pairs = await self._find_similar_pairs(
            new_nodes=new_nodes,
            agent_id=agent_id,
            max_pairs=max_pairs,
        )
        logger.debug(
            f"ContradictionResolver: {len(candidate_pairs)} candidate pairs found "
            f"for {len(new_node_ids)} new nodes (agent={agent_id})"
        )

        if not candidate_pairs:
            return ContradictionReport(candidate_pairs=0, resolved=0)

        # ── Step 3: LLM adjudication for each pair ────────────────────────────
        resolved = 0
        resolutions: list[dict[str, Any]] = []

        for new_node, existing_node in candidate_pairs[:max_pairs]:
            try:
                resolution = await self._adjudicate(new_node, existing_node)
                if resolution and resolution.get("is_contradiction"):
                    await self._apply_resolution(resolution, new_node, existing_node)
                    resolutions.append(resolution)
                    resolved += 1
            except Exception as e:
                logger.warning(
                    f"Adjudication failed for pair "
                    f"({new_node.id}, {existing_node.id}): {e}"
                )

        return ContradictionReport(
            candidate_pairs=len(candidate_pairs),
            resolved=resolved,
            resolutions=resolutions,
        )

    async def _find_similar_pairs(
        self,
        new_nodes: list[MemoryNode],
        agent_id: str,
        max_pairs: int,
    ) -> list[tuple[MemoryNode, MemoryNode]]:
        """
        Use Qdrant vector search to find existing nodes similar to new ones.
        High cosine similarity = potential contradiction candidates.
        """
        pairs: list[tuple[MemoryNode, MemoryNode]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for new_node in new_nodes:
            # Embed the new node's content
            vector = self._embedder.encode(new_node.content, convert_to_list=True)

            # Search for similar existing nodes (same agent, different ID, semantic type)
            try:
                results = await self._qdrant.search(
                    collection_name=self._settings.qdrant_collection,
                    query_vector=vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="agent_id", match=MatchValue(value=agent_id)
                            ),
                            FieldCondition(
                                key="status", match=MatchValue(value="active")
                            ),
                            FieldCondition(
                                key="type", match=MatchValue(value="semantic")
                            ),
                        ]
                    ),
                    limit=5,
                    score_threshold=self._settings.contradiction_similarity_threshold,
                    with_payload=True,
                )
            except Exception as e:
                logger.warning(f"Qdrant search for contradictions failed: {e}")
                continue

            for result in results:
                if not result.payload:
                    continue
                existing_id = result.payload.get("node_id")
                if existing_id == new_node.id:
                    continue  # Skip self-matches

                # Deduplicate pairs (order-independent)
                pair_key = tuple(sorted([new_node.id, existing_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                existing_node = await self._neo4j.get_node(existing_id)
                if existing_node:
                    pairs.append((new_node, existing_node))

                if len(pairs) >= max_pairs:
                    return pairs

        return pairs

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=False,
    )
    async def _adjudicate(
        self,
        node_a: MemoryNode,
        node_b: MemoryNode,
    ) -> dict[str, Any] | None:
        """
        Ask the LLM micro-agent to adjudicate a potential contradiction.

        Returns the parsed JSON adjudication result, or None on failure.
        """
        user_content = f"""Fact A (id={node_a.id}):
"{node_a.content}"
Created: {node_a.created_at.isoformat()}, Confidence: {node_a.confidence:.2f}

Fact B (id={node_b.id}):
"{node_b.content}"
Created: {node_b.created_at.isoformat()}, Confidence: {node_b.confidence:.2f}

Do these facts contradict each other?"""

        llm = self._get_llm()
        response = await llm.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": _ADJUDICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=512,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        logger.debug(
            f"Adjudication result: is_contradiction={parsed.get('is_contradiction')}, "
            f"winner={parsed.get('winner_id')}"
        )
        return parsed

    async def _apply_resolution(
        self,
        resolution: dict[str, Any],
        node_a: MemoryNode,
        node_b: MemoryNode,
    ) -> None:
        """
        Apply the LLM's adjudication decision to the graph:
        - Winner gets confidence boost
        - Loser gets SUPERSEDED_BY edge and status update
        - CONTRADICTS edge is removed (or a SUPERSEDED_BY edge replaces it)
        """
        winner_id = resolution.get("winner_id")
        loser_id = resolution.get("loser_id")

        if not winner_id or not loser_id:
            logger.debug("Resolution returned no winner/loser — no graph changes")
            return

        # Boost winner confidence
        winner = node_a if node_a.id == winner_id else node_b
        await self._neo4j.update_node(
            winner_id,
            {
                "confidence": min(1.0, winner.confidence + 0.1),
            },
        )

        # Mark loser as superseded
        await self._neo4j.update_node(
            loser_id,
            {
                "status": MemoryStatus.SUPERSEDED.value,
                "salience": 0.1,  # Near-zero but preserved for audit
            },
        )

        # Create SUPERSEDED_BY edge
        superseded_edge = MemoryEdge(
            source_id=loser_id,
            target_id=winner_id,
            relation_type=RelationType.SUPERSEDED_BY,
            weight=resolution.get("confidence", 0.8),
            metadata={"reasoning": resolution.get("reasoning", "")},
        )
        await self._neo4j.create_edge(superseded_edge)

        logger.info(
            f"Contradiction resolved: {loser_id} SUPERSEDED_BY {winner_id}. "
            f"Reasoning: {resolution.get('reasoning', '')[:100]}"
        )
