"""
Tests for the Ebbinghaus Decay Engine.

Tests:
- ebbinghaus_retention: mathematical correctness
- compute_stability: scaling with access_count and salience
- new_weight_after_decay: full weight computation
- resurrection_stability: boost after reinforcement
- DecayEngine.run_decay_sweep: integration behavior (mocked Neo4j)
- Edge cases: zero elapsed time, very high elapsed time, zero stability
"""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from mnemos.memory.decay import (
    DecayEngine,
    compute_stability,
    ebbinghaus_retention,
    new_weight_after_decay,
    resurrection_stability,
)


# ─── ebbinghaus_retention ─────────────────────────────────────────────────────

class TestEbbinhausRetention:

    def test_zero_elapsed_returns_one(self):
        """At t=0, retention should be 1.0 (perfect)."""
        assert ebbinghaus_retention(0, stability=1.0) == 1.0

    def test_negative_elapsed_returns_one(self):
        """Negative elapsed time (clock skew protection) should return 1.0."""
        assert ebbinghaus_retention(-10, stability=1.0) == 1.0

    def test_exponential_decay(self):
        """Verify the formula R = e^(-t/S) directly."""
        t = 6.0
        s = 2.0
        expected = math.exp(-t / s)
        result = ebbinghaus_retention(t, s)
        assert abs(result - expected) < 1e-9

    def test_higher_stability_means_slower_decay(self):
        """Same elapsed time, higher stability should yield higher retention."""
        t = 12.0
        r_low = ebbinghaus_retention(t, stability=1.0)
        r_high = ebbinghaus_retention(t, stability=5.0)
        assert r_high > r_low

    def test_retention_at_one_stability_unit(self):
        """At t=S, retention should be e^(-1) ≈ 0.368."""
        result = ebbinghaus_retention(elapsed_hours=2.0, stability=2.0)
        expected = math.exp(-1)
        assert abs(result - expected) < 1e-9

    def test_retention_approaches_zero_at_large_t(self):
        """Very large t should yield near-zero retention."""
        result = ebbinghaus_retention(10000, stability=1.0)
        assert result < 0.001

    def test_result_always_in_0_1_range(self):
        """Retention must always be in [0, 1]."""
        for t in [0, 1, 5, 24, 100, 10000]:
            for s in [0.1, 1.0, 5.0, 50.0]:
                r = ebbinghaus_retention(t, s)
                assert 0.0 <= r <= 1.0, f"Out of range: t={t}, s={s}, r={r}"

    def test_zero_stability_returns_zero(self):
        """Zero stability means instant forgetting."""
        assert ebbinghaus_retention(1.0, stability=0.0) == 0.0

    def test_negative_stability_returns_zero(self):
        """Negative stability is degenerate — returns 0."""
        assert ebbinghaus_retention(1.0, stability=-1.0) == 0.0


# ─── compute_stability ────────────────────────────────────────────────────────

class TestComputeStability:

    def test_base_case_no_access(self):
        """With 0 accesses and salience 0, stability should equal base × 1 × 1."""
        s = compute_stability(base_stability=1.0, access_count=0, initial_salience=0.0)
        # (1 + log(1+0)) * (1 + 0) = 1 * 1 = 1.0
        assert abs(s - 1.0) < 1e-9

    def test_access_count_increases_stability(self):
        """More accesses → higher stability."""
        s0 = compute_stability(1.0, access_count=0, initial_salience=0.5)
        s5 = compute_stability(1.0, access_count=5, initial_salience=0.5)
        s20 = compute_stability(1.0, access_count=20, initial_salience=0.5)
        assert s0 < s5 < s20

    def test_higher_salience_increases_stability(self):
        """Higher initial salience → higher stability."""
        s_low = compute_stability(1.0, access_count=3, initial_salience=0.1)
        s_high = compute_stability(1.0, access_count=3, initial_salience=0.9)
        assert s_high > s_low

    def test_base_stability_scales_result(self):
        """Doubling base_stability should double the result."""
        s1 = compute_stability(1.0, access_count=5, initial_salience=0.5)
        s2 = compute_stability(2.0, access_count=5, initial_salience=0.5)
        assert abs(s2 - 2 * s1) < 1e-9

    def test_stability_always_positive(self):
        """Stability should always be > 0."""
        for count in [0, 1, 100]:
            for sal in [0.0, 0.5, 1.0]:
                s = compute_stability(1.0, access_count=count, initial_salience=sal)
                assert s > 0, f"Stability ≤ 0: count={count}, sal={sal}"


# ─── new_weight_after_decay ───────────────────────────────────────────────────

class TestNewWeightAfterDecay:

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def test_fresh_memory_unchanged(self):
        """A memory reinforced right now should keep its weight."""
        now = self._now()
        w = new_weight_after_decay(
            current_weight=0.8,
            last_reinforced=now,
            stability=2.0,
            now=now,
        )
        assert abs(w - 0.8) < 1e-6

    def test_old_memory_decays(self):
        """A memory from 48 hours ago should have lower weight."""
        now = self._now()
        old = now - timedelta(hours=48)
        w = new_weight_after_decay(0.8, last_reinforced=old, stability=2.0, now=now)
        assert w < 0.8

    def test_weight_always_in_0_1(self):
        """Decayed weight must stay in [0, 1]."""
        now = self._now()
        for hours_ago in [0, 6, 24, 168, 720]:
            past = now - timedelta(hours=hours_ago)
            w = new_weight_after_decay(1.0, past, stability=1.0, now=now)
            assert 0.0 <= w <= 1.0, f"Out of range for {hours_ago}h ago: {w}"

    def test_naive_datetime_handled(self):
        """Naive datetimes (no tzinfo) should be handled without error."""
        now = datetime.now(tz=timezone.utc)
        naive_past = datetime.utcnow() - timedelta(hours=10)  # naive
        # Should not raise
        w = new_weight_after_decay(0.9, last_reinforced=naive_past, stability=2.0, now=now)
        assert 0.0 <= w <= 1.0

    def test_high_stability_slows_decay(self):
        """Same elapsed time, higher stability → smaller weight loss."""
        now = self._now()
        past = now - timedelta(hours=24)
        w_low = new_weight_after_decay(1.0, past, stability=1.0, now=now)
        w_high = new_weight_after_decay(1.0, past, stability=10.0, now=now)
        assert w_high > w_low


# ─── resurrection_stability ──────────────────────────────────────────────────

class TestResurrectionStability:

    def test_boost_increases_stability(self):
        s_new = resurrection_stability(current_stability=1.0, boost_multiplier=2.0)
        assert s_new == 2.0

    def test_boost_multiplier_applied(self):
        s_new = resurrection_stability(current_stability=3.0, boost_multiplier=1.5)
        assert abs(s_new - 4.5) < 1e-9

    def test_large_stability_preserved(self):
        """Stability can grow indefinitely through repeated reinforcement."""
        s = 1.0
        for _ in range(10):
            s = resurrection_stability(s, boost_multiplier=2.0)
        assert s == 2 ** 10


# ─── DecayEngine (integration) ────────────────────────────────────────────────

class TestDecayEngine:

    @pytest.fixture
    def mock_neo4j(self):
        client = AsyncMock()
        client.get_active_edges_for_decay = AsyncMock(return_value=[])
        client.bulk_update_edge_weights = AsyncMock(return_value=0)
        client.bulk_archive_nodes = AsyncMock(return_value=0)
        return client

    @pytest.fixture
    def decay_engine(self, mock_neo4j):
        from mnemos.config import Settings
        settings = Settings(
            decay_threshold=0.05,
            decay_stability_base=1.0,
            decay_interval_hours=6.0,
        )
        return DecayEngine(neo4j_client=mock_neo4j, settings=settings)

    @pytest.mark.asyncio
    async def test_empty_graph_sweep_returns_zero_counts(
        self, decay_engine, mock_neo4j
    ):
        mock_neo4j.get_active_edges_for_decay = AsyncMock(return_value=[])
        result = await decay_engine.run_decay_sweep()
        assert result["updated"] == 0
        assert result["archived"] == 0
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_sweep_updates_edge_weights(self, decay_engine, mock_neo4j):
        """A single old edge should get a lower weight after the sweep."""
        old_time = (datetime.now(tz=timezone.utc) - timedelta(hours=100)).isoformat()
        mock_neo4j.get_active_edges_for_decay = AsyncMock(
            side_effect=[
                [
                    {
                        "source_id": "node_a",
                        "target_id": "node_b",
                        "rel_type": "RELATES_TO",
                        "weight": 0.9,
                        "last_reinforced": old_time,
                        "edge_id": "edge_001",
                    }
                ],
                [],  # Second call returns empty → stops loop
            ]
        )
        mock_neo4j.bulk_update_edge_weights = AsyncMock(return_value=1)
        mock_neo4j.bulk_archive_nodes = AsyncMock(return_value=0)

        result = await decay_engine.run_decay_sweep()
        assert result["updated"] == 1
        mock_neo4j.bulk_update_edge_weights.assert_called_once()

        # Check the new weight is lower than original 0.9
        call_args = mock_neo4j.bulk_update_edge_weights.call_args
        updates = call_args[1]["updates"] if "updates" in call_args[1] else call_args[0][0]
        assert updates[0]["new_weight"] < 0.9

    @pytest.mark.asyncio
    async def test_below_threshold_archives_node(self, decay_engine, mock_neo4j):
        """Edge weight below DECAY_THRESHOLD should trigger node archiving."""
        very_old_time = (
            datetime.now(tz=timezone.utc) - timedelta(hours=10000)
        ).isoformat()
        mock_neo4j.get_active_edges_for_decay = AsyncMock(
            side_effect=[
                [
                    {
                        "source_id": "node_a",
                        "target_id": "node_b",
                        "rel_type": "RELATES_TO",
                        "weight": 0.9,
                        "last_reinforced": very_old_time,
                        "edge_id": "edge_001",
                    }
                ],
                [],
            ]
        )
        mock_neo4j.bulk_update_edge_weights = AsyncMock(return_value=1)
        mock_neo4j.bulk_archive_nodes = AsyncMock(return_value=1)

        result = await decay_engine.run_decay_sweep()
        # Very old edge should trigger archiving
        mock_neo4j.bulk_archive_nodes.assert_called_once()
        assert result["archived"] == 1

    @pytest.mark.asyncio
    async def test_missing_timestamp_skips_decay(self, decay_engine, mock_neo4j):
        """Edges with no last_reinforced timestamp should not be decayed."""
        mock_neo4j.get_active_edges_for_decay = AsyncMock(
            side_effect=[
                [
                    {
                        "source_id": "node_a",
                        "target_id": "node_b",
                        "rel_type": "RELATES_TO",
                        "weight": 0.8,
                        "last_reinforced": None,
                        "edge_id": "edge_002",
                    }
                ],
                [],
            ]
        )
        mock_neo4j.bulk_update_edge_weights = AsyncMock(return_value=1)
        mock_neo4j.bulk_archive_nodes = AsyncMock(return_value=0)

        await decay_engine.run_decay_sweep()

        # Weight should remain 0.8 (unchanged) since no timestamp
        call_args = mock_neo4j.bulk_update_edge_weights.call_args
        updates = call_args[1].get("updates") or call_args[0][0]
        assert updates[0]["new_weight"] == 0.8

    @pytest.mark.asyncio
    async def test_resurrect_memory_updates_node(self, decay_engine, mock_neo4j):
        """Resurrection should flip status back to active and boost stability."""
        from mnemos.models import MemoryNode, MemoryType, MemoryStatus
        from datetime import datetime, timezone

        mock_node = MemoryNode(
            id="node_archived",
            type=MemoryType.SEMANTIC,
            agent_id="test-agent",
            content="Archived fact",
            salience=0.1,
            stability=1.0,
            status=MemoryStatus.ARCHIVED,
        )
        mock_neo4j.get_node = AsyncMock(return_value=mock_node)
        mock_neo4j.update_node = AsyncMock(return_value=True)

        await decay_engine.resurrect_memory("node_archived", reinforcement_weight=0.8)

        mock_neo4j.update_node.assert_called_once()
        call_kwargs = mock_neo4j.update_node.call_args[0]
        updates = call_kwargs[1]
        assert updates["status"] == "active"
        assert updates["stability"] > 1.0  # Should be boosted
        assert updates["salience"] > 0.1   # Should be increased
