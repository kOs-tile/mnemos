"""
Async Neo4j driver wrapper for MNEMOS.

Provides:
- Schema initialization (constraints + indexes)
- CRUD for MemoryNode and MemoryEdge
- Graph traversal up to N hops
- Bulk operations for efficient ingestion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ConstraintError, Neo4jError

from mnemos.config import Settings, get_settings
from mnemos.graph.schema import (
    MEMORY_TYPE_LABELS,
    SCHEMA_QUERIES,
    EdgeProp,
    NodeLabel,
    NodeProp,
    RelLabel,
)
from mnemos.models import MemoryEdge, MemoryNode, MemoryStatus, MemoryType


def _serialize_datetime(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string for Neo4j."""
    return dt.isoformat()


def _parse_datetime(value: Any) -> datetime:
    """Parse a Neo4j datetime value to Python datetime."""
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_native"):
        return value.to_native().replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def _node_from_record(record: dict[str, Any]) -> MemoryNode:
    """Convert a Neo4j node record dict to a MemoryNode."""
    return MemoryNode(
        id=record[NodeProp.ID],
        type=MemoryType(record[NodeProp.TYPE]),
        content=record[NodeProp.CONTENT],
        agent_id=record[NodeProp.AGENT_ID],
        session_id=record.get(NodeProp.SESSION_ID),
        salience=float(record.get(NodeProp.SALIENCE, 0.5)),
        confidence=float(record.get(NodeProp.CONFIDENCE, 1.0)),
        access_count=int(record.get(NodeProp.ACCESS_COUNT, 0)),
        stability=float(record.get(NodeProp.STABILITY, 1.0)),
        status=MemoryStatus(record.get(NodeProp.STATUS, "active")),
        source_trace_id=record.get(NodeProp.SOURCE_TRACE_ID),
        created_at=_parse_datetime(record[NodeProp.CREATED_AT]),
        last_accessed=_parse_datetime(record[NodeProp.LAST_ACCESSED]),
        metadata=json.loads(record.get(NodeProp.METADATA, "{}") or "{}"),
    )


def _edge_from_record(record: dict[str, Any], relation_type: str) -> MemoryEdge:
    """Convert a Neo4j relationship record to a MemoryEdge."""
    from mnemos.models import RelationType
    return MemoryEdge(
        id=record.get(EdgeProp.ID, ""),
        source_id=record.get("source_id", ""),
        target_id=record.get("target_id", ""),
        relation_type=RelationType(relation_type),
        weight=float(record.get(EdgeProp.WEIGHT, 1.0)),
        confidence=float(record.get(EdgeProp.CONFIDENCE, 1.0)),
        created_at=_parse_datetime(record[EdgeProp.CREATED_AT]),
        last_reinforced=_parse_datetime(record[EdgeProp.LAST_REINFORCED]),
        metadata=json.loads(record.get(EdgeProp.METADATA, "{}") or "{}"),
    )


class Neo4jClient:
    """
    Async Neo4j driver wrapper.

    Usage:
        client = Neo4jClient()
        await client.connect()
        await client.init_schema()
        node = await client.create_node(memory_node)
        await client.close()

    Or as async context manager:
        async with Neo4jClient() as client:
            ...
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize the async Neo4j driver."""
        if self._driver is not None:
            return
        self._driver = AsyncGraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            max_connection_pool_size=self._settings.neo4j_max_connection_pool_size,
            connection_timeout=self._settings.neo4j_connection_timeout,
        )
        logger.info(f"Neo4j driver initialized: {self._settings.neo4j_uri}")

    async def close(self) -> None:
        """Close the driver and release connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    async def __aenter__(self) -> Neo4jClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _session(self) -> AsyncSession:
        if self._driver is None:
            raise RuntimeError("Neo4jClient not connected. Call .connect() first.")
        return self._driver.session()

    # ── Schema ────────────────────────────────────────────────────────────────

    async def init_schema(self) -> None:
        """Create all constraints and indexes defined in schema.py."""
        async with self._session() as session:
            for query in SCHEMA_QUERIES:
                try:
                    await session.run(query)
                    logger.debug(f"Schema query applied: {query[:60]}...")
                except Neo4jError as e:
                    logger.warning(f"Schema query warning (may already exist): {e.message}")
        logger.info("Neo4j schema initialized")

    # ── Node CRUD ─────────────────────────────────────────────────────────────

    async def create_node(self, node: MemoryNode) -> MemoryNode:
        """
        Create a MemoryNode in Neo4j.
        Uses MERGE on id to be idempotent.
        Applies the correct type label (Episodic/Semantic/Procedural).
        """
        type_label = MEMORY_TYPE_LABELS.get(node.type, NodeLabel.SEMANTIC)
        query = (
            f"MERGE (m:{NodeLabel.MEMORY} {{id: $id}}) "
            f"ON CREATE SET m:{type_label}, "
            f"  m.type = $type, "
            f"  m.content = $content, "
            f"  m.agent_id = $agent_id, "
            f"  m.session_id = $session_id, "
            f"  m.salience = $salience, "
            f"  m.confidence = $confidence, "
            f"  m.access_count = $access_count, "
            f"  m.stability = $stability, "
            f"  m.status = $status, "
            f"  m.source_trace_id = $source_trace_id, "
            f"  m.created_at = $created_at, "
            f"  m.last_accessed = $last_accessed, "
            f"  m.metadata = $metadata "
            f"ON MATCH SET "
            f"  m.salience = $salience, "
            f"  m.access_count = m.access_count + 1, "
            f"  m.last_accessed = $last_accessed "
            f"RETURN m"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                id=node.id,
                type=node.type,
                content=node.content,
                agent_id=node.agent_id,
                session_id=node.session_id,
                salience=node.salience,
                confidence=node.confidence,
                access_count=node.access_count,
                stability=node.stability,
                status=node.status,
                source_trace_id=node.source_trace_id,
                created_at=_serialize_datetime(node.created_at),
                last_accessed=_serialize_datetime(node.last_accessed),
                metadata=json.dumps(node.metadata),
            )
            record = await result.single()
            logger.debug(f"Created/merged node {node.id} type={node.type}")
            return node

    async def get_node(self, node_id: str) -> MemoryNode | None:
        """Fetch a single MemoryNode by id."""
        query = f"MATCH (m:{NodeLabel.MEMORY} {{id: $id}}) RETURN m"
        async with self._session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record is None:
                return None
            return _node_from_record(dict(record["m"]))

    async def update_node(self, node_id: str, updates: dict[str, Any]) -> bool:
        """Partial update of a MemoryNode's properties."""
        set_clauses = ", ".join(f"m.{k} = ${k}" for k in updates)
        query = f"MATCH (m:{NodeLabel.MEMORY} {{id: $id}}) SET {set_clauses} RETURN m.id"
        async with self._session() as session:
            result = await session.run(query, id=node_id, **updates)
            record = await result.single()
            return record is not None

    async def delete_node(self, node_id: str) -> bool:
        """Delete a MemoryNode and all its relationships."""
        query = (
            f"MATCH (m:{NodeLabel.MEMORY} {{id: $id}}) "
            "DETACH DELETE m RETURN count(m) AS deleted"
        )
        async with self._session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            deleted = record["deleted"] if record else 0
            return deleted > 0

    async def bulk_create_nodes(self, nodes: list[MemoryNode]) -> int:
        """Create multiple nodes in a single transaction. Returns count created."""
        if not nodes:
            return 0

        params_list = [
            {
                "id": n.id,
                "type": n.type,
                "content": n.content,
                "agent_id": n.agent_id,
                "session_id": n.session_id,
                "salience": n.salience,
                "confidence": n.confidence,
                "access_count": n.access_count,
                "stability": n.stability,
                "status": n.status,
                "source_trace_id": n.source_trace_id,
                "created_at": _serialize_datetime(n.created_at),
                "last_accessed": _serialize_datetime(n.last_accessed),
                "metadata": json.dumps(n.metadata),
                "type_label": MEMORY_TYPE_LABELS.get(n.type, NodeLabel.SEMANTIC),
            }
            for n in nodes
        ]

        query = (
            "UNWIND $nodes AS props "
            f"MERGE (m:{NodeLabel.MEMORY} {{id: props.id}}) "
            "ON CREATE SET m += props "
            "RETURN count(m) AS created"
        )
        async with self._session() as session:
            result = await session.run(query, nodes=params_list)
            record = await result.single()
            count = record["created"] if record else 0
            logger.info(f"Bulk created {count} nodes")
            return count

    # ── Edge CRUD ─────────────────────────────────────────────────────────────

    async def create_edge(self, edge: MemoryEdge) -> MemoryEdge:
        """
        Create a typed, weighted edge between two MemoryNodes.
        MERGE on (source)-[rel_type]->(target) to be idempotent.
        """
        query = (
            f"MATCH (s:{NodeLabel.MEMORY} {{id: $source_id}}) "
            f"MATCH (t:{NodeLabel.MEMORY} {{id: $target_id}}) "
            f"MERGE (s)-[r:{edge.relation_type}]->(t) "
            "ON CREATE SET "
            f"  r.id = $id, "
            f"  r.weight = $weight, "
            f"  r.confidence = $confidence, "
            f"  r.created_at = $created_at, "
            f"  r.last_reinforced = $last_reinforced, "
            f"  r.metadata = $metadata "
            "ON MATCH SET "
            f"  r.weight = ($weight + r.weight) / 2.0, "
            f"  r.last_reinforced = $last_reinforced "
            "RETURN r"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                id=edge.id,
                weight=edge.weight,
                confidence=edge.confidence,
                created_at=_serialize_datetime(edge.created_at),
                last_reinforced=_serialize_datetime(edge.last_reinforced),
                metadata=json.dumps(edge.metadata),
            )
            record = await result.single()
            if record is None:
                logger.warning(
                    f"Edge creation failed — nodes may not exist: "
                    f"{edge.source_id} -[{edge.relation_type}]-> {edge.target_id}"
                )
            else:
                logger.debug(
                    f"Created edge {edge.source_id} -[{edge.relation_type}]-> {edge.target_id}"
                )
            return edge

    async def get_edges_for_node(
        self, node_id: str, direction: str = "both"
    ) -> tuple[list[MemoryEdge], list[MemoryEdge]]:
        """
        Get outgoing and incoming edges for a node.
        direction: 'out' | 'in' | 'both'
        """
        outgoing: list[MemoryEdge] = []
        incoming: list[MemoryEdge] = []

        if direction in ("out", "both"):
            q_out = (
                f"MATCH (s:{NodeLabel.MEMORY} {{id: $id}})-[r]->(t:{NodeLabel.MEMORY}) "
                "RETURN type(r) AS rel_type, properties(r) AS props, "
                "s.id AS source_id, t.id AS target_id"
            )
            async with self._session() as session:
                result = await session.run(q_out, id=node_id)
                async for record in result:
                    props = dict(record["props"])
                    props["source_id"] = record["source_id"]
                    props["target_id"] = record["target_id"]
                    outgoing.append(_edge_from_record(props, record["rel_type"]))

        if direction in ("in", "both"):
            q_in = (
                f"MATCH (s:{NodeLabel.MEMORY})-[r]->(t:{NodeLabel.MEMORY} {{id: $id}}) "
                "RETURN type(r) AS rel_type, properties(r) AS props, "
                "s.id AS source_id, t.id AS target_id"
            )
            async with self._session() as session:
                result = await session.run(q_in, id=node_id)
                async for record in result:
                    props = dict(record["props"])
                    props["source_id"] = record["source_id"]
                    props["target_id"] = record["target_id"]
                    incoming.append(_edge_from_record(props, record["rel_type"]))

        return outgoing, incoming

    # ── Graph Traversal ───────────────────────────────────────────────────────

    async def traverse(
        self,
        start_node_ids: list[str],
        max_hops: int = 2,
        agent_id: str | None = None,
        min_weight: float = 0.0,
    ) -> list[MemoryNode]:
        """
        BFS/variable-depth traversal from seed nodes.
        Returns all reachable MemoryNodes within max_hops.
        """
        if not start_node_ids:
            return []

        agent_filter = "AND n.agent_id = $agent_id" if agent_id else ""
        query = (
            f"MATCH path = (start:{NodeLabel.MEMORY})-[r*1..{max_hops}]-(n:{NodeLabel.MEMORY}) "
            f"WHERE start.id IN $start_ids "
            f"  AND n.status = 'active' "
            f"  AND ALL(rel IN relationships(path) WHERE coalesce(rel.weight, 1.0) >= $min_weight) "
            f"  {agent_filter} "
            f"RETURN DISTINCT n "
            f"LIMIT 200"
        )
        params: dict[str, Any] = {
            "start_ids": start_node_ids,
            "min_weight": min_weight,
        }
        if agent_id:
            params["agent_id"] = agent_id

        nodes: list[MemoryNode] = []
        async with self._session() as session:
            result = await session.run(query, **params)
            async for record in result:
                try:
                    nodes.append(_node_from_record(dict(record["n"])))
                except Exception as e:
                    logger.warning(f"Failed to parse traversal node: {e}")

        logger.debug(
            f"Traversal from {len(start_node_ids)} seeds, depth={max_hops}: "
            f"{len(nodes)} nodes found"
        )
        return nodes

    async def find_nodes_by_content_similarity(
        self,
        node_ids: list[str],
        agent_id: str,
        limit: int = 50,
    ) -> list[MemoryNode]:
        """
        Fetch MemoryNodes by their IDs (used after Qdrant vector search).
        """
        if not node_ids:
            return []
        query = (
            f"MATCH (m:{NodeLabel.MEMORY}) "
            "WHERE m.id IN $ids AND m.agent_id = $agent_id AND m.status = 'active' "
            "RETURN m ORDER BY m.salience DESC LIMIT $limit"
        )
        nodes: list[MemoryNode] = []
        async with self._session() as session:
            result = await session.run(query, ids=node_ids, agent_id=agent_id, limit=limit)
            async for record in result:
                try:
                    nodes.append(_node_from_record(dict(record["m"])))
                except Exception as e:
                    logger.warning(f"Failed to parse node from IDs: {e}")
        return nodes

    async def get_active_edges_for_decay(
        self, batch_size: int = 500, offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        Fetch edges with their last_reinforced timestamp for decay computation.
        Returns lightweight dicts (avoids full model parsing overhead at scale).
        """
        query = (
            "MATCH (s)-[r]->(t) "
            "WHERE r.weight IS NOT NULL AND r.weight > 0 "
            "RETURN s.id AS source_id, t.id AS target_id, "
            "  type(r) AS rel_type, r.weight AS weight, "
            "  r.last_reinforced AS last_reinforced, r.last_decay AS last_decay, "
            "  coalesce(t.access_count, 0) AS target_access_count, "
            "  coalesce(t.salience, r.weight) AS target_salience, "
            "  t.stability AS target_stability, r.id AS edge_id "
            "SKIP $offset LIMIT $batch_size"
        )
        rows: list[dict[str, Any]] = []
        async with self._session() as session:
            result = await session.run(query, offset=offset, batch_size=batch_size)
            async for record in result:
                rows.append(dict(record))
        return rows

    async def bulk_update_edge_weights(
        self, updates: list[dict[str, Any]]
    ) -> int:
        """
        Bulk update edge weights after decay computation.
        Each update dict: {source_id, target_id, rel_type, new_weight, new_status?}
        """
        if not updates:
            return 0

        query = (
            "UNWIND $updates AS u "
            "MATCH (s {id: u.source_id})-[r]->(t {id: u.target_id}) "
            "WHERE type(r) = u.rel_type "
            "SET r.weight = u.new_weight, r.last_decay = datetime() "
            "RETURN count(r) AS updated"
        )
        async with self._session() as session:
            result = await session.run(query, updates=updates)
            record = await result.single()
            return record["updated"] if record else 0

    async def bulk_archive_nodes(self, node_ids: list[str]) -> int:
        """Mark nodes as archived (below decay threshold)."""
        if not node_ids:
            return 0
        query = (
            f"MATCH (m:{NodeLabel.MEMORY}) WHERE m.id IN $ids "
            "SET m.status = 'archived' RETURN count(m) AS archived"
        )
        async with self._session() as session:
            result = await session.run(query, ids=node_ids)
            record = await result.single()
            return record["archived"] if record else 0

    async def find_candidate_contradictions(
        self,
        node_ids: list[str],
        agent_id: str,
        limit: int = 20,
    ) -> list[tuple[MemoryNode, MemoryNode]]:
        """
        Find pairs of SEMANTIC nodes owned by agent_id that share
        high-confidence content but haven't yet been resolved.
        node_ids are the newly-created candidate nodes to check against existing graph.
        """
        if not node_ids:
            return []

        query = (
            f"MATCH (new:{NodeLabel.MEMORY}:{NodeLabel.SEMANTIC}) "
            f"MATCH (existing:{NodeLabel.MEMORY}:{NodeLabel.SEMANTIC}) "
            "WHERE new.id IN $new_ids "
            "  AND existing.agent_id = $agent_id "
            "  AND existing.status = 'active' "
            "  AND NOT (new)-[:CONTRADICTS|SUPERSEDED_BY]-(existing) "
            "  AND new.id <> existing.id "
            "RETURN new, existing LIMIT $limit"
        )
        pairs: list[tuple[MemoryNode, MemoryNode]] = []
        async with self._session() as session:
            result = await session.run(
                query, new_ids=node_ids, agent_id=agent_id, limit=limit
            )
            async for record in result:
                try:
                    new_node = _node_from_record(dict(record["new"]))
                    existing_node = _node_from_record(dict(record["existing"]))
                    pairs.append((new_node, existing_node))
                except Exception as e:
                    logger.warning(f"Failed to parse contradiction candidate: {e}")
        return pairs

    async def get_agent_memory_stats(self, agent_id: str) -> dict[str, Any]:
        """Return aggregate stats for an agent's memory profile."""
        query = (
            f"MATCH (m:{NodeLabel.MEMORY} {{agent_id: $agent_id}}) "
            "RETURN "
            "  count(m) AS total, "
            "  sum(CASE WHEN m.type = 'episodic' THEN 1 ELSE 0 END) AS episodic, "
            "  sum(CASE WHEN m.type = 'semantic' THEN 1 ELSE 0 END) AS semantic, "
            "  sum(CASE WHEN m.type = 'procedural' THEN 1 ELSE 0 END) AS procedural, "
            "  avg(m.salience) AS avg_salience, "
            "  min(m.created_at) AS oldest, "
            "  max(m.created_at) AS newest"
        )
        async with self._session() as session:
            result = await session.run(query, agent_id=agent_id)
            record = await result.single()
            if not record:
                return {}
            return dict(record)

    async def get_top_nodes_by_salience(
        self, agent_id: str, limit: int = 10
    ) -> list[MemoryNode]:
        """Get the highest-salience active nodes for an agent."""
        query = (
            f"MATCH (m:{NodeLabel.MEMORY} {{agent_id: $agent_id, status: 'active'}}) "
            "RETURN m ORDER BY m.salience DESC LIMIT $limit"
        )
        nodes: list[MemoryNode] = []
        async with self._session() as session:
            result = await session.run(query, agent_id=agent_id, limit=limit)
            async for record in result:
                try:
                    nodes.append(_node_from_record(dict(record["m"])))
                except Exception as e:
                    logger.warning(f"Failed to parse top node: {e}")
        return nodes

    async def mark_accessed(self, node_ids: list[str]) -> None:
        """Update access_count and last_accessed for retrieved nodes."""
        if not node_ids:
            return
        now = _serialize_datetime(datetime.now(tz=timezone.utc))
        query = (
            f"MATCH (m:{NodeLabel.MEMORY}) WHERE m.id IN $ids "
            "SET m.access_count = m.access_count + 1, m.last_accessed = $now"
        )
        async with self._session() as session:
            await session.run(query, ids=node_ids, now=now)
