"""
Neo4j schema constants — node labels, relationship types, property names.

These constants are the single source of truth for the graph schema.
Import from here instead of hard-coding strings in queries.
"""

from __future__ import annotations

# ─── Node Labels ──────────────────────────────────────────────────────────────

class NodeLabel:
    """Neo4j node labels."""

    MEMORY = "Memory"           # Base label on all memory nodes
    EPISODIC = "Episodic"       # Subtype: agent execution traces
    SEMANTIC = "Semantic"       # Subtype: extracted facts/entities
    PROCEDURAL = "Procedural"   # Subtype: skill execution patterns
    AGENT = "Agent"             # Agent profile node


# ─── Relationship Types ───────────────────────────────────────────────────────

class RelLabel:
    """Neo4j relationship type labels."""

    RELATES_TO = "RELATES_TO"
    CAUSED_BY = "CAUSED_BY"
    PRECEDED_BY = "PRECEDED_BY"
    INSTANCE_OF = "INSTANCE_OF"
    USES_PROCEDURE = "USES_PROCEDURE"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
    PART_OF = "PART_OF"
    SIMILAR_TO = "SIMILAR_TO"
    OWNED_BY = "OWNED_BY"           # Memory → Agent


# ─── Property Keys ────────────────────────────────────────────────────────────

class NodeProp:
    """Property keys for Memory nodes."""

    ID = "id"
    TYPE = "type"
    CONTENT = "content"
    AGENT_ID = "agent_id"
    SESSION_ID = "session_id"
    SALIENCE = "salience"
    CONFIDENCE = "confidence"
    ACCESS_COUNT = "access_count"
    STABILITY = "stability"
    STATUS = "status"
    SOURCE_TRACE_ID = "source_trace_id"
    CREATED_AT = "created_at"
    LAST_ACCESSED = "last_accessed"
    METADATA = "metadata"


class EdgeProp:
    """Property keys for relationship edges."""

    ID = "id"
    WEIGHT = "weight"
    CONFIDENCE = "confidence"
    CREATED_AT = "created_at"
    LAST_REINFORCED = "last_reinforced"
    METADATA = "metadata"


# ─── Cypher Schema Initialization ─────────────────────────────────────────────

SCHEMA_QUERIES: list[str] = [
    # Uniqueness constraints
    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS "
    "FOR (m:Memory) REQUIRE m.id IS UNIQUE",

    "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS "
    "FOR (a:Agent) REQUIRE a.id IS UNIQUE",

    # Indexes for common lookups
    "CREATE INDEX memory_agent_id IF NOT EXISTS "
    "FOR (m:Memory) ON (m.agent_id)",

    "CREATE INDEX memory_type IF NOT EXISTS "
    "FOR (m:Memory) ON (m.type)",

    "CREATE INDEX memory_status IF NOT EXISTS "
    "FOR (m:Memory) ON (m.status)",

    "CREATE INDEX memory_salience IF NOT EXISTS "
    "FOR (m:Memory) ON (m.salience)",

    "CREATE INDEX memory_created_at IF NOT EXISTS "
    "FOR (m:Memory) ON (m.created_at)",

    "CREATE INDEX memory_last_accessed IF NOT EXISTS "
    "FOR (m:Memory) ON (m.last_accessed)",

    # Full-text index for content search
    "CREATE FULLTEXT INDEX memory_content_fts IF NOT EXISTS "
    "FOR (m:Memory) ON EACH [m.content]",
]

# Labels that identify an active memory (excludes archived)
ACTIVE_LABELS = f":{NodeLabel.MEMORY} {{status: 'active'}}"

# All memory type labels as a list
MEMORY_TYPE_LABELS: dict[str, str] = {
    "episodic": NodeLabel.EPISODIC,
    "semantic": NodeLabel.SEMANTIC,
    "procedural": NodeLabel.PROCEDURAL,
}
