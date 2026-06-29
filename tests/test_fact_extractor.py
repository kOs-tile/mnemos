"""
Tests for FactExtractor — LLM structured output parsing.

Uses mock LLM responses to test:
- Happy path: well-formed JSON → FactExtraction
- Partial response: some fields missing → graceful defaults
- Malformed JSON: → fallback FactExtraction
- Relation validation: auto-insert missing entities
- Batch extraction: multiple traces
"""

from __future__ import annotations

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mnemos.memory.fact_extractor import FactExtractor, _trace_to_prompt
from mnemos.models import (
    AgentTrace,
    ExtractedEntity,
    ExtractedRelation,
    FactExtraction,
    MemoryType,
    ToolCall,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_trace() -> AgentTrace:
    return AgentTrace(
        agent_id="test-agent",
        session_id="sess_test_001",
        inputs={"user_message": "What does AWS EC2 t3.medium cost per month?"},
        tool_calls=[
            ToolCall(
                tool="calculator",
                args={"expression": "0.0416 * 730"},
                result=30.368,
            ),
            ToolCall(
                tool="web_search",
                args={"query": "AWS EC2 t3.medium price 2026"},
                result="t3.medium: $0.0416/hour on-demand in us-east-1",
            ),
        ],
        outputs={
            "response": "AWS EC2 t3.medium costs $30.37/month "
                        "($0.0416/hour × 730 hours) in us-east-1."
        },
        errors=[],
        duration_ms=1200,
    )


@pytest.fixture
def well_formed_extraction_json() -> str:
    return json.dumps({
        "entities": [
            {
                "name": "AWS EC2 t3.medium",
                "type": "cloud_resource",
                "description": "AWS EC2 instance type, 2 vCPU, 4GB RAM",
                "memory_type": "semantic",
                "confidence": 0.92,
            },
            {
                "name": "us-east-1",
                "type": "aws_region",
                "description": "AWS US East (N. Virginia) region",
                "memory_type": "semantic",
                "confidence": 0.99,
            },
            {
                "name": "AWS cost calculation",
                "type": "procedure",
                "description": "Multiply hourly rate by 730 to get monthly cost",
                "memory_type": "procedural",
                "confidence": 0.95,
            },
        ],
        "relations": [
            {
                "source_entity": "AWS EC2 t3.medium",
                "target_entity": "us-east-1",
                "relation_type": "priced_in",
                "description": "Instance pricing is region-specific",
                "weight": 0.90,
                "confidence": 0.92,
            }
        ],
        "summary": "Calculated monthly cost of AWS EC2 t3.medium as $30.37 in us-east-1.",
        "procedural_insight": "Monthly EC2 cost = hourly_rate × 730 hours",
        "error_patterns": [],
    })


def _make_mock_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestFactExtractorHappyPath:
    """Test successful extraction from well-formed LLM responses."""

    @pytest.mark.asyncio
    async def test_extract_returns_fact_extraction(
        self, sample_trace, well_formed_extraction_json
    ):
        extractor = FactExtractor()
        mock_response = _make_mock_openai_response(well_formed_extraction_json)

        with patch.object(
            extractor._get_client(),
            "chat",
            create=True,
        ):
            # Patch the full create call
            with patch(
                "mnemos.memory.fact_extractor.AsyncOpenAI"
            ) as MockOpenAI:
                mock_client = AsyncMock()
                MockOpenAI.return_value = mock_client
                mock_client.chat.completions.create = AsyncMock(
                    return_value=mock_response
                )
                extractor._client = mock_client

                result = await extractor.extract(sample_trace)

        assert isinstance(result, FactExtraction)
        assert len(result.entities) == 3
        assert len(result.relations) == 1
        assert result.summary != ""
        assert result.procedural_insight == "Monthly EC2 cost = hourly_rate × 730 hours"

    @pytest.mark.asyncio
    async def test_extracted_entities_have_correct_types(
        self, sample_trace, well_formed_extraction_json
    ):
        extractor = FactExtractor()
        mock_response = _make_mock_openai_response(well_formed_extraction_json)

        with patch("mnemos.memory.fact_extractor.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            extractor._client = mock_client

            result = await extractor.extract(sample_trace)

        entity_types = {e.memory_type for e in result.entities}
        assert MemoryType.SEMANTIC in entity_types
        assert MemoryType.PROCEDURAL in entity_types

    @pytest.mark.asyncio
    async def test_entity_names_populated(
        self, sample_trace, well_formed_extraction_json
    ):
        extractor = FactExtractor()
        mock_response = _make_mock_openai_response(well_formed_extraction_json)

        with patch("mnemos.memory.fact_extractor.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            extractor._client = mock_client

            result = await extractor.extract(sample_trace)

        names = {e.name for e in result.entities}
        assert "AWS EC2 t3.medium" in names
        assert "us-east-1" in names


class TestFactExtractorFallback:
    """Test fallback behavior on malformed or partial LLM responses."""

    @pytest.mark.asyncio
    async def test_malformed_json_returns_fallback(self, sample_trace):
        """Should return a minimal FactExtraction, not raise an exception."""
        extractor = FactExtractor()
        mock_response = _make_mock_openai_response("This is not JSON at all!!")

        with patch("mnemos.memory.fact_extractor.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            extractor._client = mock_client

            result = await extractor.extract(sample_trace)

        assert isinstance(result, FactExtraction)
        assert result.entities == []
        assert result.relations == []
        assert "Agent test-agent" in result.summary or result.summary != ""

    @pytest.mark.asyncio
    async def test_empty_response_returns_fallback(self, sample_trace):
        """Should handle empty LLM response gracefully."""
        extractor = FactExtractor()
        mock_response = _make_mock_openai_response(None)

        with patch("mnemos.memory.fact_extractor.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            extractor._client = mock_client

            result = await extractor.extract(sample_trace)

        assert isinstance(result, FactExtraction)

    @pytest.mark.asyncio
    async def test_partial_json_missing_entities_key(self, sample_trace):
        """JSON missing 'entities' key should fill in defaults."""
        extractor = FactExtractor()
        partial_json = json.dumps({"summary": "Something happened", "relations": []})
        mock_response = _make_mock_openai_response(partial_json)

        with patch("mnemos.memory.fact_extractor.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            MockOpenAI.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            extractor._client = mock_client

            result = await extractor.extract(sample_trace)

        assert isinstance(result, FactExtraction)
        assert result.summary == "Something happened"


class TestFactExtractionModel:
    """Test FactExtraction Pydantic model validation."""

    def test_relation_entity_auto_inserted(self):
        """
        If a relation references an entity name not in the entities list,
        the model validator should auto-insert a placeholder entity.
        """
        extraction = FactExtraction(
            entities=[
                ExtractedEntity(
                    name="AWS EC2",
                    type="service",
                    description="Cloud compute service",
                    memory_type=MemoryType.SEMANTIC,
                )
            ],
            relations=[
                ExtractedRelation(
                    source_entity="AWS EC2",
                    target_entity="us-east-1",  # Not in entities!
                    relation_type="priced_in",
                )
            ],
            summary="Test summary",
        )

        # Validator should have auto-inserted 'us-east-1'
        entity_names = {e.name for e in extraction.entities}
        assert "us-east-1" in entity_names
        assert len(extraction.entities) == 2

    def test_empty_extraction_is_valid(self):
        """Empty extraction should not raise validation errors."""
        extraction = FactExtraction()
        assert extraction.entities == []
        assert extraction.relations == []
        assert extraction.summary == ""

    def test_error_patterns_populated(self):
        extraction = FactExtraction(
            entities=[],
            relations=[],
            summary="Ran into 429 rate limit from OpenAI.",
            error_patterns=["OpenAI 429 rate limit", "retry after 60s"],
        )
        assert len(extraction.error_patterns) == 2


class TestTraceToPrompt:
    """Test _trace_to_prompt helper function."""

    def test_includes_agent_id(self, sample_trace):
        prompt = _trace_to_prompt(sample_trace)
        assert "test-agent" in prompt

    def test_includes_inputs(self, sample_trace):
        prompt = _trace_to_prompt(sample_trace)
        assert "AWS EC2 t3.medium" in prompt

    def test_includes_tool_calls(self, sample_trace):
        prompt = _trace_to_prompt(sample_trace)
        assert "calculator" in prompt
        assert "web_search" in prompt

    def test_includes_outputs(self, sample_trace):
        prompt = _trace_to_prompt(sample_trace)
        assert "30.37" in prompt

    def test_empty_tool_calls(self):
        trace = AgentTrace(
            agent_id="a",
            inputs={"q": "test"},
            tool_calls=[],
            outputs={"r": "ok"},
        )
        prompt = _trace_to_prompt(trace)
        assert "none" in prompt.lower() or "(none)" in prompt
