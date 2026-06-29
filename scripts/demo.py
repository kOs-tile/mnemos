"""
MNEMOS Interactive Demo

Demonstrates:
1. Ingesting fake agent traces via the SDK
2. Querying memory with natural language
3. Observing how salience changes over access
4. Triggering the decay engine manually
5. Showing contradiction detection in action

Usage:
    # With MNEMOS running (docker-compose up -d):
    python scripts/demo.py

    # With mock mode (no live service):
    python scripts/demo.py --mock
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger


# ─── Demo Traces ─────────────────────────────────────────────────────────────

DEMO_TRACES = [
    {
        "agent_id": "demo-agent",
        "session_id": "demo-session-001",
        "inputs": {"user_message": "What's the AWS cost for running 5 t3.medium instances?"},
        "tool_calls": [
            {
                "tool": "calculator",
                "args": {"expression": "0.0416 * 730 * 5"},
                "result": 151.84,
            }
        ],
        "outputs": {
            "response": "Running 5 AWS t3.medium instances costs approximately $151.84/month "
                        "(based on $0.0416/hour × 730 hours × 5 instances)."
        },
        "errors": [],
        "duration_ms": 892,
    },
    {
        "agent_id": "demo-agent",
        "session_id": "demo-session-002",
        "inputs": {"user_message": "Who founded Stripe and when?"},
        "tool_calls": [
            {
                "tool": "web_search",
                "args": {"query": "Stripe founders history"},
                "result": "Stripe was founded in 2010 by Patrick Collison and John Collison.",
            }
        ],
        "outputs": {
            "response": "Stripe was founded in 2010 by brothers Patrick Collison and John Collison."
        },
        "errors": [],
        "duration_ms": 1240,
    },
    {
        "agent_id": "demo-agent",
        "session_id": "demo-session-003",
        "inputs": {"user_message": "Schedule a meeting with the backend team for code review"},
        "tool_calls": [
            {
                "tool": "google_calendar",
                "args": {
                    "title": "Backend Code Review",
                    "date": "2026-07-01",
                    "time": "10:00",
                },
                "result": {"event_id": "cal_xyz789", "status": "created"},
            }
        ],
        "outputs": {"response": "Meeting scheduled for July 1, 2026 at 10:00 AM."},
        "errors": [],
        "duration_ms": 654,
    },
    {
        "agent_id": "demo-agent",
        "session_id": "demo-session-004",
        "inputs": {"user_message": "Write a Python function to retry failed API calls"},
        "tool_calls": [
            {
                "tool": "code_writer",
                "args": {"language": "python", "task": "retry decorator with exponential backoff"},
                "result": "def retry(max_attempts=3, backoff=1.5): ...",
            }
        ],
        "outputs": {
            "response": "Here's a retry decorator using exponential backoff...",
            "code_snippet": "from tenacity import retry, stop_after_attempt, wait_exponential",
        },
        "errors": [],
        "duration_ms": 2100,
    },
    # Contradicting fact (same entity, different value)
    {
        "agent_id": "demo-agent",
        "session_id": "demo-session-005",
        "inputs": {"user_message": "What's the latest AWS t3.medium price?"},
        "tool_calls": [
            {
                "tool": "web_search",
                "args": {"query": "AWS EC2 t3.medium us-east-1 price 2026"},
                "result": "AWS EC2 t3.medium: $0.0416/hour in us-east-1 (current pricing page)",
            }
        ],
        "outputs": {
            "response": "The current price for AWS EC2 t3.medium in us-east-1 is $0.0416/hour."
        },
        "errors": [],
        "duration_ms": 980,
    },
]

DEMO_QUERIES = [
    "What do we know about AWS pricing?",
    "Tell me about Stripe",
    "How do I retry failed API calls?",
    "What meetings have been scheduled?",
    "What companies were researched?",
]


# ─── Mock Mode (no live service) ─────────────────────────────────────────────

class MockMnemosClient:
    """Mock client for demo without a live MNEMOS service."""

    def __init__(self, agent_id: str = "demo-agent"):
        self.agent_id = agent_id
        self._memories: list[dict] = []

    async def ingest_trace(self, trace) -> object:
        self._memories.append({
            "content": str(trace.outputs),
            "type": "episodic",
            "salience": 0.75,
        })

        class FakeResponse:
            trace_id = trace.trace_id
            nodes_created = 3
            edges_created = 2
            contradictions_found = 0
            processing_ms = 150

        return FakeResponse()

    async def query(self, query: str, top_k: int = 5, **kwargs) -> object:
        class FakeQuery:
            memories = []
            formatted_context = f"[Mock context for: '{query}'] — No live MNEMOS service."
            retrieval_ms = 5

        return FakeQuery()

    async def close(self):
        pass


# ─── Demo Runner ─────────────────────────────────────────────────────────────

async def run_demo(mock: bool = False, mnemos_url: str = "http://localhost:8000"):
    print("\n" + "═" * 70)
    print("  MNEMOS — Persistent Semantic Memory Graph Demo")
    print("═" * 70 + "\n")

    if mock:
        print("⚠  Running in MOCK MODE — no live service required\n")
        from mnemos.models import AgentTrace
        client = MockMnemosClient(agent_id="demo-agent")
    else:
        from mnemos.sdk.client import MnemosClient
        from mnemos.models import AgentTrace
        client = MnemosClient(base_url=mnemos_url, agent_id="demo-agent")
        # Verify connection
        try:
            health = await client.health_check()
            print(f"✓ Connected to MNEMOS at {mnemos_url}")
            print(f"  Neo4j: {health.get('neo4j', 'unknown')}")
            print(f"  Qdrant: {health.get('qdrant', 'unknown')}")
            print(f"  Redis: {health.get('redis', 'unknown')}\n")
        except Exception as e:
            print(f"✗ Cannot connect to MNEMOS at {mnemos_url}: {e}")
            print("  Run: docker-compose up -d")
            print("  Or:  python scripts/demo.py --mock\n")
            return

    # ── Phase 1: Ingest traces ────────────────────────────────────────────────
    print("─" * 70)
    print("PHASE 1: Ingesting Agent Traces")
    print("─" * 70)

    for i, trace_data in enumerate(DEMO_TRACES, 1):
        trace = AgentTrace(**trace_data)
        print(f"\n[{i}/{len(DEMO_TRACES)}] Ingesting: {trace.inputs['user_message'][:60]}...")

        t0 = time.time()
        result = await client.ingest_trace(trace)
        elapsed = int((time.time() - t0) * 1000)

        print(f"  ✓ trace_id={result.trace_id[:8]}...")
        print(f"    nodes_created={result.nodes_created}, "
              f"edges_created={result.edges_created}, "
              f"contradictions={result.contradictions_found}")
        print(f"    processing={result.processing_ms}ms (round-trip={elapsed}ms)")

        # Small delay to simulate real agent cadence
        await asyncio.sleep(0.5)

    # ── Phase 2: Query memory ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 2: Querying Memory")
    print("─" * 70)

    for query in DEMO_QUERIES:
        print(f"\nQuery: '{query}'")
        print("  Retrieving...", end="", flush=True)

        t0 = time.time()
        result = await client.query(query, top_k=3, include_graph_hops=2)
        elapsed = int((time.time() - t0) * 1000)

        print(f" done ({elapsed}ms, {len(result.memories)} memories)\n")
        print(result.formatted_context)

    # ── Phase 3: Show memory profile ─────────────────────────────────────────
    if not mock:
        print("\n" + "─" * 70)
        print("PHASE 3: Agent Memory Profile")
        print("─" * 70)
        try:
            profile = await client.get_agent_profile()
            print(f"\nAgent: {profile.agent_id}")
            print(f"  Total nodes:    {profile.total_nodes}")
            print(f"  Episodic:       {profile.episodic_count}")
            print(f"  Semantic:       {profile.semantic_count}")
            print(f"  Procedural:     {profile.procedural_count}")
            print(f"  Avg salience:   {profile.avg_salience:.3f}")
            if profile.top_entities:
                print("\n  Top memories by salience:")
                for m in profile.top_entities[:5]:
                    print(f"    [{m.salience:.2f}] [{m.type}] {m.content[:70]}...")
        except Exception as e:
            print(f"  (skipped — {e})")

    # ── Phase 4: Decay simulation ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 4: Ebbinghaus Decay Simulation")
    print("─" * 70)

    from mnemos.memory.decay import ebbinghaus_retention, compute_stability
    print("\nRetention curve for a memory (stability=1.0):")
    print(f"  {'Hours':>8}  {'Retention':>12}  {'Visual'}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*30}")

    for hours in [0, 1, 6, 12, 24, 48, 72, 168]:
        retention = ebbinghaus_retention(hours, stability=1.0)
        bar = "█" * int(retention * 30)
        label = f"{hours}h" if hours < 168 else "7 days"
        print(f"  {label:>8}  {retention:>11.1%}  {bar}")

    print("\nWith access_count=5 (stability×3.0 via reinforcement):")
    print(f"  {'Hours':>8}  {'Retention':>12}  {'Visual'}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*30}")
    for hours in [0, 24, 72, 168, 720]:
        s = compute_stability(base_stability=1.0, access_count=5, initial_salience=0.8)
        retention = ebbinghaus_retention(hours, stability=s)
        bar = "█" * int(retention * 30)
        label = f"{hours}h" if hours < 720 else "30 days"
        print(f"  {label:>8}  {retention:>11.1%}  {bar}")

    await client.close()

    print("\n" + "═" * 70)
    print("  Demo complete!")
    print(f"  Neo4j Browser:  http://localhost:7474")
    print(f"  MNEMOS Docs:    {mnemos_url}/docs")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNEMOS Interactive Demo")
    parser.add_argument("--mock", action="store_true", help="Run without live MNEMOS service")
    parser.add_argument(
        "--url",
        default=os.getenv("MNEMOS_URL", "http://localhost:8000"),
        help="MNEMOS service URL",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(mock=args.mock, mnemos_url=args.url))
