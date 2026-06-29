"""
Memory Ingestion Pipeline.

Receives an AgentTrace and:
1. Calls FactExtractor to get structured entities/relations
2. Creates MemoryNodes in Neo4j for each extracted entity
3. Creates MemoryEdges for each extracted relation
4. Creates an Episodic node for the trace itself
5. Embeds node content into Qdrant for vector search
6. Triggers ContradictionResolver for newly created semantic nodes
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

from mnemos.config import Settings, get_settings
from mnemos.graph.neo4j_client import Neo4jClient
from mnemos.graph.schema import NodeLabel, RelLabel
from mnemos.memory.contradiction import ContradictionResolver
from mnemos.memory.fact_extractor import FactExtractor
from mnemos.models import (
    AgentTrace,
    ExtractedEntity,
    ExtractedRelation,
    FactExtraction,
    IngestResponse,
    MemoryEdge,
    MemoryNode,
    MemoryType,
    RelationType,
)


def _map_relation_type(raw: str) -> RelationType:
    """Map a free-text relation label to a typed RelationType."""
    mapping: dict[str, RelationType] = {
        "relates to": RelationType.RELATES_TO,
        "relates_to": RelationType.RELATES_TO,
        "caused by": RelationType.CAUSED_BY,
        "caused_by": RelationType.CAUSED_BY,
        "preceded by": RelationType.PRECEDED_BY,
        "preceded_by": RelationType.PRECEDED_BY,
        "instance of": RelationType.INSTANCE_OF,
        "instance_of": RelationType.INSTANCE_OF,
        "uses procedure": RelationType.USES_PROCEDURE,
        "uses_procedure": RelationType.USES_PROCEDURE,
        "has attribute": RelationType.HAS_ATTRIBUTE,
        "has_attribute": RelationType.HAS_ATTRIBUTE,
        "part of": RelationType.PART_OF,
        "part_of": RelationType.PART_OF,
        "similar to": RelationType.SIMILAR_TO,
        "similar_to": RelationType.SIMILAR_TO,
    }
    normalized = raw.lower().strip()
    return mapping.get(normalized, RelationType.RELATES_TO)


class IngestionPipeline:
    """
    Orchestrates the full memory ingestion pipeline from AgentTrace to stored graph.

    Usage:
        pipeline = IngestionPipeline()
        await pipeline.initialize()
        response = await pipeline.ingest(trace)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._neo4j: Neo4jClient | None = None
        self._qdrant: AsyncQdrantClient | None = None
        self._embedder: SentenceTransformer | None = None
        self._extractor: FactExtractor | None = None
        self._contradiction_resolver: ContradictionResolver | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Connect to all backends and load the embedding model."""
        if self._initialized:
            return

        logger.info("Initializing IngestionPipeline...")

        self._neo4j = Neo4jClient(self._settings)
        await self._neo4j.connect()
        await self._neo4j.init_schema()

        self._qdrant = AsyncQdrantClient(
            host=self._settings.qdrant_host,
            port=self._settings.qdrant_port,
        )
        await self._ensure_qdrant_collection()

        logger.info(f"Loading embedding model: {self._settings.embedding_model}")
        self._embedder = SentenceTransformer(self._settings.embedding_model)

        self._extractor = FactExtractor(self._settings)
        self._contradiction_resolver = ContradictionResolver(
            neo4j_client=self._neo4j,
            qdrant_client=self._qdrant,
            embedder=self._embedder,
            settings=self._settings,
        )

        self._initialized = True
        logger.info("IngestionPipeline ready")

    async def _ensure_qdrant_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        try:
            await self._qdrant.get_collection(self._settings.qdrant_collection)
            logger.debug(f"Qdrant collection '{self._settings.qdrant_collection}' exists")
        except Exception:
            logger.info(
                f"Creating Qdrant collection '{self._settings.qdrant_collection}' "
                f"dim={self._settings.embedding_dim}"
            )
            await self._qdrant.create_collection(
                collection_name=self._settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=self._settings.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    async def ingest(
        self, trace: AgentTrace, force_extract: bool = False
    ) -> IngestResponse:
        """
        Full ingestion pipeline for one AgentTrace.

        Steps:
        1. Extract facts (LLM)
        2. Create Episodic node for the trace
        3. Create Semantic/Procedural nodes for each extracted entity
        4. Create edges between nodes
        5. Embed all new nodes into Qdrant
        6. Run contradiction detection
        """
        if not self._initialized:
            await self.initialize()

        start_ms = int(time.time() * 1000)
        nodes_created = 0
        edges_created = 0
        contradictions_found = 0

        # ── Step 1: Extract facts ─────────────────────────────────────────────
        extraction: FactExtraction = await self._extractor.extract(trace)

        # ── Step 2: Create Episodic trace node ────────────────────────────────
        trace_node = MemoryNode(
            type=MemoryType.EPISODIC,
            content=extraction.summary or (
                f"Agent {trace.agent_id} executed with inputs: "
                f"{str(trace.inputs)[:200]}"
            ),
            agent_id=trace.agent_id,
            session_id=trace.session_id,
            salience=0.7,
            source_trace_id=trace.trace_id,
        )
        await self._neo4j.create_node(trace_node)
        nodes_created += 1

        # ── Step 3: Create entity nodes ───────────────────────────────────────
        entity_name_to_node: dict[str, MemoryNode] = {}

        for entity in extraction.entities:
            memory_type = MemoryType(entity.memory_type)

            node = MemoryNode(
                type=memory_type,
                content=f"{entity.name}: {entity.description}",
                agent_id=trace.agent_id,
                session_id=trace.session_id,
                salience=entity.confidence,
                confidence=entity.confidence,
                source_trace_id=trace.trace_id,
                metadata={
                    "entity_name": entity.name,
                    "entity_type": entity.type,
                    **entity.metadata,
                },
            )
            await self._neo4j.create_node(node)
            entity_name_to_node[entity.name] = node
            nodes_created += 1

            # Link entity node back to trace node
            link_edge = MemoryEdge(
                source_id=trace_node.id,
                target_id=node.id,
                relation_type=RelationType.INSTANCE_OF
                if memory_type == MemoryType.SEMANTIC
                else RelationType.RELATES_TO,
                weight=entity.confidence,
            )
            await self._neo4j.create_edge(link_edge)
            edges_created += 1

        # ── Step 4: Handle procedural insight ────────────────────────────────
        if extraction.procedural_insight:
            proc_node = MemoryNode(
                type=MemoryType.PROCEDURAL,
                content=extraction.procedural_insight,
                agent_id=trace.agent_id,
                session_id=trace.session_id,
                salience=0.8,
                source_trace_id=trace.trace_id,
            )
            await self._neo4j.create_node(proc_node)
            nodes_created += 1

            proc_edge = MemoryEdge(
                source_id=trace_node.id,
                target_id=proc_node.id,
                relation_type=RelationType.USES_PROCEDURE,
                weight=0.9,
            )
            await self._neo4j.create_edge(proc_edge)
            edges_created += 1

        # ── Step 5: Create relation edges ─────────────────────────────────────
        for relation in extraction.relations:
            source_node = entity_name_to_node.get(relation.source_entity)
            target_node = entity_name_to_node.get(relation.target_entity)
            if not source_node or not target_node:
                logger.warning(
                    f"Skipping relation {relation.source_entity} → {relation.target_entity}: "
                    "one or both nodes not found"
                )
                continue

            rel_type = _map_relation_type(relation.relation_type)
            edge = MemoryEdge(
                source_id=source_node.id,
                target_id=target_node.id,
                relation_type=rel_type,
                weight=relation.weight,
                confidence=relation.confidence,
                metadata={"description": relation.description},
            )
            await self._neo4j.create_edge(edge)
            edges_created += 1

        # ── Step 6: Embed all new nodes into Qdrant ───────────────────────────
        all_new_nodes = [trace_node] + list(entity_name_to_node.values())
        await self._embed_nodes(all_new_nodes)

        # ── Step 7: Contradiction detection ───────────────────────────────────
        semantic_node_ids = [
            n.id
            for n in entity_name_to_node.values()
            if n.type == MemoryType.SEMANTIC
        ]
        if semantic_node_ids:
            report = await self._contradiction_resolver.detect_and_resolve(
                new_node_ids=semantic_node_ids,
                agent_id=trace.agent_id,
            )
            contradictions_found = report.resolved

        end_ms = int(time.time() * 1000)
        return IngestResponse(
            trace_id=trace.trace_id,
            nodes_created=nodes_created,
            edges_created=edges_created,
            contradictions_found=contradictions_found,
            processing_ms=end_ms - start_ms,
        )

    async def _embed_nodes(self, nodes: list[MemoryNode]) -> None:
        """
        Embed node content using sentence transformer and upsert into Qdrant.
        """
        if not nodes or not self._embedder:
            return

        texts = [n.content for n in nodes]
        vectors = self._embedder.encode(texts, convert_to_list=True)

        points = [
            PointStruct(
                id=_uuid_to_int(n.id),
                vector=vec,
                payload={
                    "node_id": n.id,
                    "agent_id": n.agent_id,
                    "type": n.type,
                    "content": n.content[:500],
                    "salience": n.salience,
                    "status": n.status,
                },
            )
            for n, vec in zip(nodes, vectors)
        ]

        await self._qdrant.upsert(
            collection_name=self._settings.qdrant_collection,
            points=points,
        )
        logger.debug(f"Upserted {len(points)} vectors into Qdrant")

    async def close(self) -> None:
        """Clean up connections."""
        if self._neo4j:
            await self._neo4j.close()
        if self._qdrant:
            await self._qdrant.close()
        self._initialized = False


def _uuid_to_int(uuid_str: str) -> int:
    """Convert UUID string to integer for Qdrant point ID."""
    import uuid
    return uuid.UUID(uuid_str).int % (2**63)
