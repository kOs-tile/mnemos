# MNEMOS

> **Status — Research-active.** MNEMOS is being retained as a memory-systems experiment and KAVI subsystem candidate. Temporal-decay semantics are under active validation; do not treat the current implementation as production memory infrastructure without further workload testing.


[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1.svg)](https://neo4j.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9-red.svg)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hermes Compatible](https://img.shields.io/badge/Hermes-Compatible-8A2BE2.svg)](https://github.com/kOs-tile/hermes)

> **Provenance-aware persistent memory research for multi-agent AI systems.**
> Historical context stays advisory, freshness-bounded, and separate from runtime authority.

---

## The Memory Problem

Every time you call an LLM, it forgets everything. RAG helps with retrieval, but **vector stores aren't real memory** — they're sophisticated search indexes.

Real memory has structure:

| Property | Vector Store | MNEMOS |
|---|---|---|
| Relationships | ❌ Flat documents | ✅ Typed graph edges |
| Temporal decay | ❌ Static embeddings | ✅ Ebbinghaus forgetting curve |
| Contradiction handling | ❌ Silently conflicts | ✅ LLM micro-agent adjudication |
| Memory types | ❌ One homogeneous index | ✅ Episodic / Semantic / Procedural |
| Associative retrieval | ❌ Top-K cosine | ✅ Vector → graph traversal hybrid |
| Multi-agent isolation | ❌ Shared namespace | ✅ Per-agent scoped profiles |

MNEMOS gives your agents a **working memory** that actually behaves like one: it forgets low-salience facts, strengthens frequently accessed paths, detects contradictions, and organizes knowledge into typed memory structures.

---

## Provenance and freshness

MNEMOS records how a memory entered the system. Episodic execution records are
tagged `agent_trace`; semantic/procedural facts extracted by the LLM are tagged
`llm_derived`. The model also supports `external_evidence`, `user_asserted`,
and `unknown` provenance.

Memories may carry an optional `valid_until` boundary. Expired memories are
excluded from default retrieval even if they remain structurally ACTIVE in the
memory graph. A caller must explicitly request stale evidence to include it.

Queries may also provide a provenance allowlist. The same filter is applied to
direct vector matches and graph-expanded neighbors so graph traversal cannot
silently reintroduce disallowed memory.

Every query now includes an `admission_report` showing which candidates were
evaluated, which were rejected, the deterministic rejection reason, and whether
any policy-ineligible memory leaked into final prompt context. Evidence-sensitive
callers may set `require_source_trace=true` to fail closed on memories that cannot
be tied back to an originating AgentTrace.

## Authority boundary

MNEMOS returns advisory historical context, not execution authority. Prompt-formatted retrieval explicitly warns that memories can be stale, incorrect, or superseded and must not override current task/policy. Where available, retrieved items include the originating trace identifier for provenance.

In a KAVI stack, KCC remains the authority plane; a remembered instruction cannot grant a capability that is absent from the current execution capsule.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HERMES AGENT FRAMEWORK                    │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│   │  Skill A     │  │  Skill B     │  │  Orchestrator        │  │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│          │   HermesHook    │                      │              │
└──────────┼─────────────────┼──────────────────────┼─────────────┘
           │                 │  AgentTrace           │
           ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MNEMOS FastAPI Service                       │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│  │  /ingest    │  │  /query     │  │  /agent/{id}/memory  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────────────────┘    │
│         │                │                                       │
│  ┌──────▼──────┐  ┌──────▼───────────────────────────────┐     │
│  │ FactExtract │  │           QueryPlanner                │     │
│  │  (LLM call) │  │  embed → Qdrant → Neo4j traversal    │     │
│  └──────┬──────┘  └──────────────────────────────────────┘     │
│         │                                                        │
│  ┌──────▼──────────────────────────────────┐                    │
│  │         ContradictionResolver            │                    │
│  │  candidate pruning → LLM adjudication   │                    │
│  └──────┬──────────────────────────────────┘                    │
│         │                                                        │
└─────────┼────────────────────────────────────────────────────── ┘
          │
     ┌────▼────────────────────────────────────────────────┐
     │                  Storage Layer                       │
     │                                                      │
     │  ┌──────────────────┐  ┌──────────────────────┐    │
     │  │     Neo4j 5.x    │  │      Qdrant 1.9       │    │
     │  │  Memory Graph    │  │  Vector Embeddings    │    │
     │  │                  │  │                       │    │
     │  │  (Episodic)──────│  │  384-dim sentence    │    │
     │  │  (Semantic)      │  │  transformer vecs    │    │
     │  │  (Procedural)    │  │                       │    │
     │  └──────────────────┘  └──────────────────────┘    │
     │                                                      │
     │  ┌──────────────────┐                               │
     │  │     Redis         │                               │
     │  │  Agent sessions  │                               │
     │  │  Query cache      │                               │
     │  └──────────────────┘                               │
     └──────────────────────────────────────────────────────┘
          │
     ┌────▼────────────────────────────────────────────────┐
     │              APScheduler (Decay Engine)              │
     │                                                      │
     │  Every 6h: R(t) = e^(-t/S) × salience_weight       │
     │  Resurrect edges when reinforced above threshold    │
     └──────────────────────────────────────────────────────┘
```

---

## Memory Types

MNEMOS models three distinct memory types, mirroring cognitive science:

### Episodic Memory
**"What happened"** — Concrete events, agent execution traces, timestamped interactions.

```
(AgentRun:Episodic {
    content: "User asked about AWS costs, tool_call=calculator, result=$1,240/mo",
    salience: 0.82,
    agent_id: "kavi-orchestrator",
    created_at: "2026-01-15T14:23:00Z"
})
```

### Semantic Memory
**"What is true"** — Extracted facts, entity relationships, world knowledge.

```
(Entity:Semantic {
    content: "AWS EC2 t3.medium costs $0.0416/hour in us-east-1",
    salience: 0.95,
    confidence: 0.88
})
-[:RELATES_TO {weight: 0.9}]->
(Entity:Semantic { content: "AWS pricing" })
```

### Procedural Memory
**"How to do it"** — Successful skill execution patterns, tool usage strategies.

```
(Procedure:Procedural {
    content: "To calculate AWS costs: use calculator tool with hourly_rate × 730",
    salience: 0.75,
    success_count: 14
})
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- OpenAI or DeepSeek API key

### 1. Clone & Configure

```bash
git clone https://github.com/kOs-tile/mnemos.git
cd mnemos
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Services

```bash
docker-compose up -d
```

This starts Neo4j (bolt://localhost:7687), Qdrant (http://localhost:6333), Redis (localhost:6379), and the MNEMOS API (http://localhost:8000).

### 3. Seed Demo Data

```bash
pip install -r requirements.txt
python scripts/seed_demo_data.py
```

### 4. Run Interactive Demo

```bash
python scripts/demo.py
```

---

## Python SDK Usage

### Installation

```bash
pip install mnemos-sdk  # or: pip install -e .
```

### Basic Usage

```python
from mnemos.sdk.client import MnemosClient
from mnemos.models import AgentTrace

client = MnemosClient(base_url="http://localhost:8000", agent_id="my-agent")

# Ingest an agent execution trace
trace = AgentTrace(
    agent_id="my-agent",
    session_id="sess_abc123",
    inputs={"user_message": "What's the capital of France?"},
    tool_calls=[
        {"tool": "web_search", "args": {"query": "capital of France"}, "result": "Paris"}
    ],
    outputs={"response": "The capital of France is Paris."},
    errors=[],
    duration_ms=1240
)
await client.ingest_trace(trace)

# Query memory with natural language
context = await client.query(
    "What do we know about France?",
    top_k=5,
    include_graph_hops=2
)
print(context.memories)
# [MemoryNode(content="The capital of France is Paris", salience=0.91, ...)]

# Get a specific entity
entity = await client.get_entity("entity_paris_001")
print(entity.relations)  # Graph neighbors
```

### Async Context Manager

```python
async with MnemosClient(base_url="http://localhost:8000", agent_id="my-agent") as client:
    result = await client.query("AWS cost estimates we've calculated")
    for memory in result.memories:
        print(f"[{memory.type}] {memory.content} (salience={memory.salience:.2f})")
```

---

## Hermes Integration

Drop the `HermesHook` into any Hermes skill for zero-config memory ingestion:

```python
from mnemos.sdk.hermes_hook import HermesHook

# Wrap a skill execution
hook = HermesHook(
    mnemos_url="http://localhost:8000",
    agent_id="kavi-orchestrator"
)

@hook.trace_skill("web_search")
async def web_search_skill(query: str) -> dict:
    # Your existing skill logic
    result = await do_web_search(query)
    return {"result": result}

# MNEMOS automatically captures inputs, outputs, timing, errors
# and ingests structured memory after each skill call
```

### Orchestrator-Level Integration

```python
from mnemos.sdk.hermes_hook import HermesHook
from mnemos.sdk.client import MnemosClient

class KaviOrchestrator:
    def __init__(self):
        self.memory = MnemosClient(
            base_url=os.getenv("MNEMOS_URL"),
            agent_id="kavi-orchestrator"
        )
        self.hook = HermesHook(client=self.memory)

    async def run(self, user_message: str) -> str:
        # Retrieve relevant memory before planning
        context = await self.memory.query(user_message, top_k=10)

        # Inject memory context into system prompt
        memory_context = context.format_for_prompt()
        plan = await self.planner.plan(user_message, context=memory_context)

        # Execute skills with auto-tracing
        async with self.hook.session(session_id=plan.session_id):
            result = await plan.execute()

        return result
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Ingest an AgentTrace |
| `POST` | `/query` | Natural language memory query |
| `GET` | `/entity/{id}` | Get entity + graph neighbors |
| `POST` | `/agent/{agent_id}/memory` | Get agent's full memory profile |
| `DELETE` | `/memory/{id}` | Delete a memory node |
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | OpenAPI docs (Swagger UI) |

---

## Decay Engine

MNEMOS implements the **Ebbinghaus forgetting curve**:

```
R(t) = e^(-t / S)
```

Where `R` is retention (0–1), `t` is time elapsed since last access, and `S` is the stability factor (scales with salience and reinforcement count).

- Edges with `weight < decay_threshold` (default `0.05`) are marked `archived`
- When a memory is accessed again, it gets **resurrected** with a new stability factor
- APScheduler runs the decay sweep every 6 hours

---

## Contradiction Resolution

When new facts conflict with existing memory:

1. **Candidate pruning** — embedding similarity search finds potentially conflicting nodes
2. **LLM adjudication** — a micro-agent (DeepSeek/GPT-4o) receives both facts + provenance
3. **Graph update** — winning fact gets `confidence += 0.1`, loser gets `superseded_by` edge

```python
# Automatic — triggered on every /ingest call
# Manual resolution:
await client.resolve_contradictions(entity_id="entity_aws_price_001")
```

---

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `OPENAI_API_KEY` | — | OpenAI key (or use DeepSeek) |
| `DEEPSEEK_API_KEY` | — | DeepSeek key |
| `LLM_PROVIDER` | `openai` | `openai` or `deepseek` |
| `DECAY_INTERVAL_HOURS` | `6` | Decay sweep interval |
| `DECAY_THRESHOLD` | `0.05` | Archive threshold |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |

---

## Project Structure

```
mnemos/
├── mnemos/
│   ├── __init__.py
│   ├── config.py              # Pydantic Settings
│   ├── models.py              # Core Pydantic models
│   ├── graph/
│   │   ├── neo4j_client.py    # Async Neo4j driver wrapper
│   │   └── schema.py          # Node labels, relationship types
│   ├── memory/
│   │   ├── ingestion.py       # AgentTrace → graph pipeline
│   │   ├── fact_extractor.py  # LLM structured output extraction
│   │   ├── retrieval.py       # QueryPlanner: vector + graph hybrid
│   │   ├── decay.py           # Ebbinghaus decay engine
│   │   └── contradiction.py   # ContradictionResolver
│   ├── api/
│   │   ├── routes.py          # FastAPI route handlers
│   │   └── middleware.py      # Agent identity scoping
│   └── sdk/
│       ├── client.py          # MnemosClient Python SDK
│       └── hermes_hook.py     # Hermes skill auto-tracing hook
├── scripts/
│   ├── demo.py                # Interactive demo
│   └── seed_demo_data.py      # Sample data population
├── tests/
│   ├── test_fact_extractor.py
│   ├── test_decay.py
│   └── test_retrieval.py
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with hot reload
uvicorn mnemos.api.routes:app --reload --port 8000

# Check Neo4j schema
python -c "from mnemos.graph.neo4j_client import Neo4jClient; import asyncio; asyncio.run(Neo4jClient().init_schema())"
```

---

## License

MIT License — Copyright (c) 2026 Onur Kavi

---

*MNEMOS — from the Greek μνήμη (mneme), meaning memory. In Greek mythology, Mnemosyne was the goddess of memory and mother of the nine Muses.*


## Validation gate

MNEMOS's primary safety metric is **silent-trust rate**: stale or disallowed memory must not enter default prompt context without explicit opt-in. The benchmark and exit gate are defined in [`docs/VALIDATION.md`](docs/VALIDATION.md).

The original six-case admission smoke benchmark remains executable with `python benchmark/admission.py`. The broader trust-boundary benchmark runs with `python -m benchmark.admission_matrix`.

Current CI checkpoint: **24/24 trust-boundary checks match expected outcomes with 0 policy leaks** — 20 direct admission-policy cases plus 4 QueryPlanner graph-expansion cases. The graph cases verify that stale, disallowed-provenance, and missing-source-trace neighbors do not silently enter final prompt context, while an allowed neighbor remains retrievable.

This validates the admission contract only; it is not a substitute for LoCoMo/LongMemEval/BEAM-style memory-quality evaluation or a production memory-quality score.
