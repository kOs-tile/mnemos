"""
seed_demo_data.py — Populate Neo4j with a realistic sample memory graph.

Creates a rich demo dataset for screenshots and interactive demos:
- Multiple agents with scoped memories
- Episodic/Semantic/Procedural nodes
- Typed edges
- Some intentional contradictions to show resolution

Run: python scripts/seed_demo_data.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger

from mnemos.config import get_settings
from mnemos.graph.neo4j_client import Neo4jClient
from mnemos.models import MemoryEdge, MemoryNode, MemoryStatus, MemoryType, RelationType


AGENT_IDS = ["kavi-orchestrator", "research-agent", "code-agent"]


SAMPLE_NODES = [
    # ── Kavi Orchestrator ─────────────────────────────────────────────────────
    MemoryNode(
        id="ep_001",
        type=MemoryType.EPISODIC,
        agent_id="kavi-orchestrator",
        session_id="sess_demo_001",
        content="User asked for a competitor analysis of Notion vs Linear. "
                "Used web_search tool 4 times, returned 800-word report.",
        salience=0.85,
        confidence=0.95,
        access_count=3,
    ),
    MemoryNode(
        id="ep_002",
        type=MemoryType.EPISODIC,
        agent_id="kavi-orchestrator",
        session_id="sess_demo_002",
        content="AWS cost calculation for production workload: t3.medium × 10 instances. "
                "Calculated $412/month using EC2 pricing API.",
        salience=0.78,
        confidence=0.92,
        access_count=5,
    ),
    MemoryNode(
        id="ep_003",
        type=MemoryType.EPISODIC,
        agent_id="kavi-orchestrator",
        session_id="sess_demo_003",
        content="User scheduled meeting with Stripe team for Q4 API integration. "
                "Created calendar event via GCal tool.",
        salience=0.65,
        confidence=0.99,
        access_count=1,
    ),
    # ── Semantic facts ────────────────────────────────────────────────────────
    MemoryNode(
        id="sem_001",
        type=MemoryType.SEMANTIC,
        agent_id="kavi-orchestrator",
        content="AWS EC2 t3.medium costs $0.0416/hour in us-east-1 (on-demand)",
        salience=0.92,
        confidence=0.88,
        access_count=8,
        metadata={"entity_name": "AWS EC2 t3.medium", "entity_type": "pricing"},
    ),
    MemoryNode(
        id="sem_002",
        type=MemoryType.SEMANTIC,
        agent_id="kavi-orchestrator",
        content="Notion pricing: Team plan is $8/user/month (2026), "
                "includes unlimited blocks and version history",
        salience=0.75,
        confidence=0.82,
        access_count=4,
        metadata={"entity_name": "Notion", "entity_type": "saas_product"},
    ),
    MemoryNode(
        id="sem_003",
        type=MemoryType.SEMANTIC,
        agent_id="kavi-orchestrator",
        content="Linear is a project management tool focused on software teams. "
                "Core differentiator: keyboard-first UX and Git integration.",
        salience=0.70,
        confidence=0.90,
        access_count=3,
        metadata={"entity_name": "Linear", "entity_type": "saas_product"},
    ),
    MemoryNode(
        id="sem_004",
        type=MemoryType.SEMANTIC,
        agent_id="kavi-orchestrator",
        content="Stripe API v3 supports webhook signature verification using "
                "Stripe-Signature header with HMAC-SHA256",
        salience=0.88,
        confidence=0.95,
        access_count=6,
        metadata={"entity_name": "Stripe API", "entity_type": "api"},
    ),
    # Contradicting fact for sem_001 (to demonstrate resolution)
    MemoryNode(
        id="sem_005",
        type=MemoryType.SEMANTIC,
        agent_id="kavi-orchestrator",
        content="AWS EC2 t3.medium costs $0.0464/hour in us-east-1 (on-demand)",
        salience=0.30,
        confidence=0.55,
        access_count=1,
        status=MemoryStatus.SUPERSEDED,
        metadata={"entity_name": "AWS EC2 t3.medium", "entity_type": "pricing", "note": "outdated"},
    ),
    # ── Procedural memories ───────────────────────────────────────────────────
    MemoryNode(
        id="proc_001",
        type=MemoryType.PROCEDURAL,
        agent_id="kavi-orchestrator",
        content="To calculate monthly AWS EC2 cost: multiply hourly_rate by 730 hours. "
                "Use calculator tool with formula: hourly_rate * 730 * instance_count.",
        salience=0.95,
        confidence=0.99,
        access_count=12,
        stability=4.5,
    ),
    MemoryNode(
        id="proc_002",
        type=MemoryType.PROCEDURAL,
        agent_id="kavi-orchestrator",
        content="Competitor analysis pattern: (1) web_search for product homepage, "
                "(2) web_search for pricing page, (3) web_search for G2/Capterra reviews, "
                "(4) synthesize into structured comparison.",
        salience=0.88,
        confidence=0.92,
        access_count=7,
        stability=3.2,
    ),
    # ── Research Agent ────────────────────────────────────────────────────────
    MemoryNode(
        id="ep_ra_001",
        type=MemoryType.EPISODIC,
        agent_id="research-agent",
        session_id="sess_ra_001",
        content="Researched LLM memory architectures. Found 3 key papers: "
                "MemGPT (2023), Cognitive Architecture for Language Agents, "
                "and Retrieval-Augmented Generation (Lewis et al., 2020).",
        salience=0.88,
        confidence=0.97,
        access_count=2,
    ),
    MemoryNode(
        id="sem_ra_001",
        type=MemoryType.SEMANTIC,
        agent_id="research-agent",
        content="MemGPT (2023) proposed a hierarchical memory system for LLMs: "
                "in-context memory (main context), external memory (vector store), "
                "and archival memory (long-term storage). Key insight: LLMs as OS memory managers.",
        salience=0.90,
        confidence=0.94,
        access_count=5,
    ),
    # ── Code Agent ────────────────────────────────────────────────────────────
    MemoryNode(
        id="proc_ca_001",
        type=MemoryType.PROCEDURAL,
        agent_id="code-agent",
        content="Python async pattern for Neo4j: always use 'async with driver.session() as session:' "
                "not 'await session.close()'. Use AsyncGraphDatabase.driver(), not GraphDatabase.driver().",
        salience=0.93,
        confidence=0.98,
        access_count=9,
        stability=5.0,
    ),
]


SAMPLE_EDGES = [
    # Episodic → Semantic (instance_of)
    MemoryEdge(
        source_id="ep_002",
        target_id="sem_001",
        relation_type=RelationType.INSTANCE_OF,
        weight=0.92,
    ),
    # Episodic → Procedural (uses_procedure)
    MemoryEdge(
        source_id="ep_002",
        target_id="proc_001",
        relation_type=RelationType.USES_PROCEDURE,
        weight=0.90,
    ),
    # Semantic → Semantic (relates_to)
    MemoryEdge(
        source_id="sem_002",
        target_id="sem_003",
        relation_type=RelationType.RELATES_TO,
        weight=0.75,
        metadata={"description": "Notion and Linear are competing project management tools"},
    ),
    # Episodic → Semantic (instance_of)
    MemoryEdge(
        source_id="ep_001",
        target_id="sem_002",
        relation_type=RelationType.INSTANCE_OF,
        weight=0.80,
    ),
    MemoryEdge(
        source_id="ep_001",
        target_id="sem_003",
        relation_type=RelationType.INSTANCE_OF,
        weight=0.80,
    ),
    # Contradiction + resolution
    MemoryEdge(
        source_id="sem_005",
        target_id="sem_001",
        relation_type=RelationType.SUPERSEDED_BY,
        weight=0.88,
        metadata={"reasoning": "sem_001 uses current AWS pricing page (2026), sem_005 is from older cache"},
    ),
    # Episodic → Procedural (competitor analysis)
    MemoryEdge(
        source_id="ep_001",
        target_id="proc_002",
        relation_type=RelationType.USES_PROCEDURE,
        weight=0.85,
    ),
    # Research agent
    MemoryEdge(
        source_id="ep_ra_001",
        target_id="sem_ra_001",
        relation_type=RelationType.INSTANCE_OF,
        weight=0.94,
    ),
    # Stripe API
    MemoryEdge(
        source_id="ep_003",
        target_id="sem_004",
        relation_type=RelationType.RELATES_TO,
        weight=0.82,
    ),
]


async def seed():
    settings = get_settings()
    async with Neo4jClient(settings) as client:
        await client.init_schema()

        logger.info(f"Seeding {len(SAMPLE_NODES)} nodes...")
        for node in SAMPLE_NODES:
            await client.create_node(node)

        logger.info(f"Seeding {len(SAMPLE_EDGES)} edges...")
        for edge in SAMPLE_EDGES:
            await client.create_edge(edge)

        logger.info("✓ Demo data seeded successfully")
        logger.info(f"  Nodes: {len(SAMPLE_NODES)}")
        logger.info(f"  Edges: {len(SAMPLE_EDGES)}")
        logger.info(f"  Agents: {AGENT_IDS}")
        logger.info("\nOpen Neo4j Browser at http://localhost:7474")
        logger.info("Query: MATCH (n:Memory) RETURN n LIMIT 50")


if __name__ == "__main__":
    asyncio.run(seed())
