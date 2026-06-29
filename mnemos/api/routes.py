"""
MNEMOS FastAPI application — route definitions and lifespan management.

Routes:
    POST  /ingest                  — Ingest an AgentTrace
    POST  /query                   — Natural language memory query
    GET   /entity/{id}             — Get entity node + graph neighbors
    POST  /agent/{agent_id}/memory — Get agent memory profile
    DELETE /memory/{id}            — Delete a memory node
    GET   /health                  — Service health check

Lifespan: Initializes all backends (Neo4j, Qdrant, Redis, embedder) on startup.
          Stops decay scheduler and closes connections on shutdown.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from loguru import logger
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from mnemos.api.middleware import AgentIdentityMiddleware, RequestLoggingMiddleware
from mnemos.config import Settings, get_settings
from mnemos.graph.neo4j_client import Neo4jClient
from mnemos.memory.decay import DecayEngine
from mnemos.memory.ingestion import IngestionPipeline
from mnemos.memory.retrieval import QueryPlanner
from mnemos.models import (
    AgentMemoryProfile,
    ContradictionReport,
    EntityResponse,
    IngestRequest,
    IngestResponse,
    MemoryNode,
    MemoryType,
    QueryRequest,
    QueryResponse,
)


# ─── Application State ───────────────────────────────────────────────────────

class AppState:
    """Holds all initialized backend clients (attached to app.state)."""

    neo4j: Neo4jClient
    qdrant: AsyncQdrantClient
    redis: aioredis.Redis
    embedder: SentenceTransformer
    ingestion_pipeline: IngestionPipeline
    query_planner: QueryPlanner
    decay_engine: DecayEngine
    settings: Settings


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Initializes all backends on startup and cleans up on shutdown.
    """
    settings = get_settings()
    state = AppState()
    state.settings = settings

    logger.info("MNEMOS starting up...")

    # Neo4j
    state.neo4j = Neo4jClient(settings)
    await state.neo4j.connect()
    await state.neo4j.init_schema()
    logger.info("Neo4j connected")

    # Qdrant
    state.qdrant = AsyncQdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port
    )
    logger.info("Qdrant connected")

    # Redis
    state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    logger.info("Redis connected")

    # Sentence transformer (shared across ingestion + retrieval)
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    state.embedder = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded")

    # Ingestion pipeline (shares embedder, neo4j, qdrant)
    state.ingestion_pipeline = IngestionPipeline(settings)
    await state.ingestion_pipeline.initialize()

    # Query planner
    state.query_planner = QueryPlanner(
        neo4j_client=state.neo4j,
        qdrant_client=state.qdrant,
        embedder=state.embedder,
        settings=settings,
    )

    # Decay engine
    state.decay_engine = DecayEngine(neo4j_client=state.neo4j, settings=settings)
    state.decay_engine.start_scheduler()

    app.state.mnemos = state
    logger.info("MNEMOS ready")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("MNEMOS shutting down...")
    state.decay_engine.stop_scheduler()
    await state.ingestion_pipeline.close()
    await state.neo4j.close()
    await state.qdrant.close()
    await state.redis.aclose()
    logger.info("MNEMOS shutdown complete")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MNEMOS",
    description=(
        "Persistent semantic memory graph service for multi-agent AI systems. "
        "Designed for the Hermes/Kavi Claw agent framework."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware — order matters: logging wraps identity check
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AgentIdentityMiddleware)


# ─── Dependency: State ────────────────────────────────────────────────────────

def get_app_state(request: Request) -> AppState:
    return request.app.state.mnemos


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check(state: AppState = Depends(get_app_state)) -> dict[str, Any]:
    """Service health check. Verifies connectivity to all backends."""
    health: dict[str, Any] = {"status": "ok", "service": "mnemos", "version": "0.1.0"}

    # Neo4j
    try:
        async with state.neo4j._session() as session:
            result = await session.run("RETURN 1 AS ok")
            await result.single()
        health["neo4j"] = "ok"
    except Exception as e:
        health["neo4j"] = f"error: {e}"
        health["status"] = "degraded"

    # Qdrant
    try:
        await state.qdrant.get_collections()
        health["qdrant"] = "ok"
    except Exception as e:
        health["qdrant"] = f"error: {e}"
        health["status"] = "degraded"

    # Redis
    try:
        await state.redis.ping()
        health["redis"] = "ok"
    except Exception as e:
        health["redis"] = f"error: {e}"
        health["status"] = "degraded"

    return health


@app.post("/ingest", response_model=IngestResponse, tags=["Memory"])
async def ingest_trace(
    body: IngestRequest,
    request: Request,
    state: AppState = Depends(get_app_state),
) -> IngestResponse:
    """
    Ingest an AgentTrace into the memory graph.

    Triggers:
    1. LLM fact extraction
    2. Node + edge creation in Neo4j
    3. Vector embedding in Qdrant
    4. Contradiction detection
    """
    # Allow agent_id from header to override trace (or enforce consistency)
    header_agent_id = getattr(request.state, "agent_id", None)
    if header_agent_id and body.trace.agent_id != header_agent_id:
        body.trace.agent_id = header_agent_id

    logger.info(
        f"POST /ingest trace_id={body.trace.trace_id} agent={body.trace.agent_id}"
    )

    # Check for duplicate trace (Redis cache)
    cache_key = f"trace:{body.trace.trace_id}"
    if not body.force_extract:
        cached = await state.redis.get(cache_key)
        if cached:
            logger.debug(f"Trace {body.trace.trace_id} already processed, skipping")
            import json
            return IngestResponse(**json.loads(cached))

    response = await state.ingestion_pipeline.ingest(
        body.trace, force_extract=body.force_extract
    )

    # Cache the result to prevent duplicate processing
    await state.redis.setex(
        cache_key,
        state.settings.redis_query_cache_ttl * 10,  # 10x longer than query cache
        response.model_dump_json(),
    )

    return response


@app.post("/query", response_model=QueryResponse, tags=["Memory"])
async def query_memory(
    body: QueryRequest,
    request: Request,
    state: AppState = Depends(get_app_state),
) -> QueryResponse:
    """
    Query the memory graph with natural language.

    Returns ranked MemoryNodes with formatted prompt context.
    Supports filtering by memory type, salience threshold, and graph hops.
    """
    # Agent scoping: header takes precedence
    header_agent_id = getattr(request.state, "agent_id", None)
    if header_agent_id:
        body.agent_id = header_agent_id

    logger.info(f"POST /query agent={body.agent_id} query='{body.query[:80]}'")

    # Check Redis cache
    import hashlib
    cache_key = (
        f"query:{body.agent_id}:"
        + hashlib.md5(
            f"{body.query}{body.top_k}{body.include_graph_hops}{body.memory_types}".encode()
        ).hexdigest()[:12]
    )
    cached = await state.redis.get(cache_key)
    if cached:
        import json
        logger.debug("Returning cached query result")
        return QueryResponse(**json.loads(cached))

    response = await state.query_planner.query(body)

    # Cache for short TTL (memory changes frequently)
    await state.redis.setex(
        cache_key,
        state.settings.redis_query_cache_ttl,
        response.model_dump_json(),
    )

    return response


@app.get("/entity/{entity_id}", response_model=EntityResponse, tags=["Memory"])
async def get_entity(
    entity_id: str,
    state: AppState = Depends(get_app_state),
) -> EntityResponse:
    """
    Retrieve a specific memory node with its graph neighborhood.

    Returns the node, all outgoing/incoming edges, and immediate neighbors.
    """
    node = await state.neo4j.get_node(entity_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")

    outgoing, incoming = await state.neo4j.get_edges_for_node(entity_id, direction="both")

    # Fetch neighbor nodes
    neighbor_ids = list({e.target_id for e in outgoing} | {e.source_id for e in incoming})
    neighbors: list[MemoryNode] = []
    for nid in neighbor_ids[:20]:  # Limit to 20 neighbors
        n = await state.neo4j.get_node(nid)
        if n:
            neighbors.append(n)

    await state.neo4j.mark_accessed([entity_id])

    return EntityResponse(
        node=node,
        outgoing_edges=outgoing,
        incoming_edges=incoming,
        neighbors=neighbors,
    )


@app.post("/agent/{agent_id}/memory", response_model=AgentMemoryProfile, tags=["Agents"])
async def get_agent_memory_profile(
    agent_id: str,
    request: Request,
    state: AppState = Depends(get_app_state),
) -> AgentMemoryProfile:
    """
    Retrieve an agent's complete memory profile.

    Returns aggregate statistics and top-salience memories.
    """
    # Enforce scoping: agents can only see their own memory
    header_agent_id = getattr(request.state, "agent_id", None)
    if header_agent_id and header_agent_id != agent_id:
        raise HTTPException(
            status_code=403,
            detail="Agents can only access their own memory profile",
        )

    stats = await state.neo4j.get_agent_memory_stats(agent_id)
    if not stats or not stats.get("total"):
        raise HTTPException(
            status_code=404,
            detail=f"No memories found for agent '{agent_id}'",
        )

    top_nodes = await state.neo4j.get_top_nodes_by_salience(agent_id, limit=10)

    from datetime import datetime
    def _parse_ts(raw) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw
        if hasattr(raw, "to_native"):
            return raw.to_native()
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None

    return AgentMemoryProfile(
        agent_id=agent_id,
        total_nodes=int(stats.get("total", 0)),
        episodic_count=int(stats.get("episodic", 0)),
        semantic_count=int(stats.get("semantic", 0)),
        procedural_count=int(stats.get("procedural", 0)),
        avg_salience=float(stats.get("avg_salience") or 0.0),
        oldest_memory=_parse_ts(stats.get("oldest")),
        most_recent_memory=_parse_ts(stats.get("newest")),
        top_entities=top_nodes,
    )


@app.delete("/memory/{memory_id}", tags=["Memory"])
async def delete_memory(
    memory_id: str,
    request: Request,
    state: AppState = Depends(get_app_state),
) -> dict[str, str]:
    """
    Delete a memory node and all its relationships.

    Also removes the corresponding Qdrant vector point.
    """
    # Verify existence and ownership
    node = await state.neo4j.get_node(memory_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")

    header_agent_id = getattr(request.state, "agent_id", None)
    if header_agent_id and node.agent_id != header_agent_id:
        raise HTTPException(
            status_code=403,
            detail="Cannot delete memory belonging to another agent",
        )

    # Delete from Neo4j
    deleted = await state.neo4j.delete_node(memory_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete from Neo4j")

    # Delete from Qdrant
    try:
        from mnemos.memory.ingestion import _uuid_to_int
        await state.qdrant.delete(
            collection_name=state.settings.qdrant_collection,
            points_selector=[_uuid_to_int(memory_id)],
        )
    except Exception as e:
        logger.warning(f"Failed to delete vector for {memory_id}: {e}")

    return {"deleted": memory_id, "status": "ok"}


@app.post("/decay/run", tags=["System"])
async def trigger_decay(
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """
    Manually trigger a decay sweep (admin endpoint).
    Normally run automatically by the APScheduler.
    """
    logger.info("Manual decay sweep triggered via API")
    result = await state.decay_engine.run_decay_sweep()
    return {"sweep_result": result}
