"""
FactExtractor — LLM call with JSON schema output to extract structured entities
and relationships from AgentTrace execution records.

Supports OpenAI and DeepSeek providers. Uses structured output / JSON mode
to guarantee parseable responses.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mnemos.config import Settings, get_settings
from mnemos.models import AgentTrace, FactExtraction

# ─── System Prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a memory extraction specialist for an AI agent system.
Your job is to analyze agent execution traces and extract structured knowledge.

You will receive a JSON record of an agent's actions (inputs, tool calls, outputs, errors).
Extract:
1. **entities** — named things (people, companies, concepts, values, locations, resources)
2. **relations** — how entities relate to each other
3. **summary** — one sentence describing what this agent run accomplished
4. **procedural_insight** — if the trace shows a reusable procedure/pattern, describe it
5. **error_patterns** — any error patterns worth remembering

For memory_type, classify each entity as:
- "episodic" — specific events, timestamped facts tied to this run
- "semantic" — timeless facts, world knowledge, entity definitions
- "procedural" — how-to knowledge, patterns, workflows

Be specific. Prefer concrete facts over vague observations.
Do not repeat the same entity twice. Merge near-duplicates.
"""

_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "memory_type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "metadata": {"type": "object"},
                },
                "required": ["name", "type", "description", "memory_type"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity": {"type": "string"},
                    "target_entity": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "description": {"type": "string"},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["source_entity", "target_entity", "relation_type"],
            },
        },
        "summary": {"type": "string"},
        "procedural_insight": {"type": ["string", "null"]},
        "error_patterns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities", "relations", "summary"],
}


def _trace_to_prompt(trace: AgentTrace) -> str:
    """Format an AgentTrace as a compact string for the LLM."""
    tool_summaries = []
    for tc in trace.tool_calls:
        summary = f"  - tool={tc.tool}, args={json.dumps(tc.args)[:200]}"
        if tc.result is not None:
            summary += f", result={str(tc.result)[:300]}"
        if tc.error:
            summary += f", ERROR={tc.error}"
        tool_summaries.append(summary)

    return f"""Agent Execution Trace
====================
agent_id: {trace.agent_id}
session_id: {trace.session_id}
timestamp: {trace.timestamp.isoformat()}

INPUTS:
{json.dumps(trace.inputs, indent=2)[:500]}

TOOL CALLS ({len(trace.tool_calls)}):
{chr(10).join(tool_summaries) or "  (none)"}

OUTPUTS:
{json.dumps(trace.outputs, indent=2)[:500]}

ERRORS: {trace.errors or "none"}
"""


class FactExtractor:
    """
    Extracts structured entities and relationships from AgentTrace objects.

    Uses an LLM (OpenAI / DeepSeek) with JSON schema structured output.
    Results are validated and returned as FactExtraction Pydantic objects.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
            )
        return self._client

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def extract(self, trace: AgentTrace) -> FactExtraction:
        """
        Extract structured facts from an AgentTrace.

        Returns a FactExtraction with entities, relations, summary,
        procedural insight, and error patterns.
        """
        user_content = _trace_to_prompt(trace)
        logger.debug(f"Extracting facts from trace {trace.trace_id}")

        client = self._get_client()

        # Use JSON mode (works with both OpenAI and DeepSeek)
        response = await client.chat.completions.create(
            model=self._settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2048,
        )

        raw_content = response.choices[0].message.content or "{}"
        logger.debug(f"LLM extraction raw response: {raw_content[:200]}...")

        try:
            parsed = json.loads(raw_content)
            extraction = FactExtraction(**parsed)
            logger.info(
                f"Extracted {len(extraction.entities)} entities, "
                f"{len(extraction.relations)} relations from trace {trace.trace_id}"
            )
            return extraction
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse FactExtraction from LLM response: {e}")
            logger.debug(f"Raw LLM content: {raw_content}")
            # Return a minimal extraction with the summary at minimum
            return FactExtraction(
                entities=[],
                relations=[],
                summary=f"Agent {trace.agent_id} ran with {len(trace.tool_calls)} tool calls.",
            )

    async def extract_batch(
        self, traces: list[AgentTrace]
    ) -> list[tuple[AgentTrace, FactExtraction]]:
        """Extract facts from multiple traces sequentially."""
        results: list[tuple[AgentTrace, FactExtraction]] = []
        for trace in traces:
            extraction = await self.extract(trace)
            results.append((trace, extraction))
        return results
