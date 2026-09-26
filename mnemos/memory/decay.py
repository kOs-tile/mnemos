"""
Temporal Decay Engine — Ebbinghaus Forgetting Curve implementation.

The Ebbinghaus retention function:
    R(t) = e^(-t / S)

Where:
    R ∈ [0, 1]  — retention (current memory strength)
    t           — time elapsed since last reinforcement (in hours)
    S           — stability factor (higher = slower forgetting)
                  S scales with access_count and initial salience

Memory lifecycle:
    ACTIVE → (decay sweeps) → weight approaches 0 → ARCHIVED
    ARCHIVED → (new evidence) → weight increases → ACTIVE (resurrection)

Scheduled via APScheduler, runs every DECAY_INTERVAL_HOURS hours.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from mnemos.config import Settings, get_settings
from mnemos.graph.neo4j_client import Neo4jClient


# ─── Core Ebbinghaus Function ─────────────────────────────────────────────────


def ebbinghaus_retention(
    elapsed_hours: float,
    stability: float,
) -> float:
    """
    Compute retention R(t) using the Ebbinghaus forgetting curve.

        R(t) = e^(-t / S)

    Args:
        elapsed_hours: Hours since the memory was last reinforced.
        stability: Stability factor S. Higher = slower decay.
                   Typical range: 0.5 (unstable) to 100.0 (deeply consolidated).

    Returns:
        Retention value in [0.0, 1.0].
    """
    if elapsed_hours <= 0:
        return 1.0
    if stability <= 0:
        return 0.0
    return math.exp(-elapsed_hours / stability)


def compute_stability(
    base_stability: float,
    access_count: int,
    initial_salience: float,
) -> float:
    """
    Compute the stability factor S for a memory node.

    Stability increases with:
    - More accesses (spaced repetition effect)
    - Higher initial salience (important memories are more stable)

    Formula:
        S = base_stability × (1 + log(1 + access_count)) × (1 + initial_salience)

    Args:
        base_stability: DECAY_STABILITY_BASE config value.
        access_count: Number of times this memory has been retrieved.
        initial_salience: Salience at creation time (0–1).

    Returns:
        Stability factor S > 0.
    """
    import math
    reinforcement_factor = 1.0 + math.log1p(access_count)
    salience_factor = 1.0 + initial_salience
    return base_stability * reinforcement_factor * salience_factor


def new_weight_after_decay(
    current_weight: float,
    last_reinforced: datetime,
    stability: float,
    now: datetime | None = None,
) -> float:
    """
    Compute the new edge weight after applying temporal decay.

    Args:
        current_weight: Current edge weight (0–1).
        last_reinforced: Timestamp of last reinforcement.
        stability: Stability factor S.
        now: Current time (defaults to UTC now).

    Returns:
        New weight after decay, in [0.0, 1.0].
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)

    # Ensure both datetimes are timezone-aware
    if last_reinforced.tzinfo is None:
        last_reinforced = last_reinforced.replace(tzinfo=timezone.utc)

    elapsed_hours = (now - last_reinforced).total_seconds() / 3600.0
    retention = ebbinghaus_retention(elapsed_hours, stability)
    return current_weight * retention


def resurrection_stability(
    current_stability: float,
    boost_multiplier: float,
) -> float:
    """
    Compute new stability when an archived memory is resurrected.

    When a forgotten memory is accessed again, it recovers with
    higher stability than before (spacing effect).
    """
    return current_stability * boost_multiplier


# ─── Decay Engine ─────────────────────────────────────────────────────────────


class DecayEngine:
    """
    Scheduled decay engine that applies Ebbinghaus forgetting to the memory graph.

    Runs every DECAY_INTERVAL_HOURS hours via APScheduler.
    For each edge:
      - Computes new weight using retention formula
      - Updates Neo4j
      - Archives edges below DECAY_THRESHOLD
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        settings: Settings | None = None,
    ) -> None:
        self._neo4j = neo4j_client
        self._settings = settings or get_settings()
        self._scheduler: AsyncIOScheduler | None = None

    def start_scheduler(self) -> None:
        """Start the APScheduler background job."""
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self.run_decay_sweep,
            trigger=IntervalTrigger(hours=self._settings.decay_interval_hours),
            id="decay_sweep",
            name="Ebbinghaus Decay Sweep",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self._scheduler.start()
        logger.info(
            f"Decay scheduler started. "
            f"Interval: {self._settings.decay_interval_hours}h, "
            f"Threshold: {self._settings.decay_threshold}"
        )

    def stop_scheduler(self) -> None:
        """Stop the APScheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Decay scheduler stopped")

    async def run_decay_sweep(self) -> dict[str, int]:
        """
        Execute one full decay sweep across all edges in the graph.

        Returns a summary dict: {updated, archived, errors}
        """
        logger.info("Starting Ebbinghaus decay sweep...")
        now = datetime.now(tz=timezone.utc)
        updated_total = 0
        archived_total = 0
        errors = 0
        batch_size = 500
        offset = 0

        while True:
            batch = await self._neo4j.get_active_edges_for_decay(
                batch_size=batch_size, offset=offset
            )
            if not batch:
                break

            updates: list[dict[str, Any]] = []
            nodes_to_archive: list[str] = []

            for row in batch:
                try:
                    new_weight = self._compute_decayed_weight(row, now)
                    updates.append(
                        {
                            "source_id": row["source_id"],
                            "target_id": row["target_id"],
                            "rel_type": row["rel_type"],
                            "new_weight": new_weight,
                        }
                    )

                    # Archive if below threshold
                    if new_weight < self._settings.decay_threshold:
                        # Archive the target node (the memory that's being forgotten)
                        if row.get("target_id"):
                            nodes_to_archive.append(row["target_id"])

                except Exception as e:
                    logger.warning(f"Decay computation error for edge {row}: {e}")
                    errors += 1

            if updates:
                count = await self._neo4j.bulk_update_edge_weights(updates)
                updated_total += count

            if nodes_to_archive:
                archived = await self._neo4j.bulk_archive_nodes(
                    list(set(nodes_to_archive))
                )
                archived_total += archived

            offset += batch_size
            if len(batch) < batch_size:
                break

        summary = {
            "updated": updated_total,
            "archived": archived_total,
            "errors": errors,
        }
        logger.info(f"Decay sweep complete: {summary}")
        return summary

    def _compute_decayed_weight(
        self, edge_row: dict[str, Any], now: datetime
    ) -> float:
        """Apply Ebbinghaus decay to a single edge row from Neo4j."""
        current_weight = float(edge_row.get("weight", 1.0))
        # Decay must be incremental. Re-applying total time since reinforcement
        # to an already-decayed weight compounds the same elapsed interval on
        # every sweep. Prefer the previous decay checkpoint when available.
        reference_raw = edge_row.get("last_decay") or edge_row.get("last_reinforced")

        if reference_raw is None:
            return current_weight  # No timestamp — skip decay

        if isinstance(reference_raw, str):
            reference_time = datetime.fromisoformat(reference_raw)
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
        elif hasattr(reference_raw, "to_native"):
            reference_time = reference_raw.to_native().replace(tzinfo=timezone.utc)
        else:
            reference_time = reference_raw

        stored_stability = edge_row.get("target_stability")
        if stored_stability is not None and float(stored_stability) > 0:
            stability = float(stored_stability)
        else:
            stability = compute_stability(
                base_stability=self._settings.decay_stability_base,
                access_count=int(edge_row.get("target_access_count") or 0),
                initial_salience=float(edge_row.get("target_salience") or current_weight),
            )

        return new_weight_after_decay(
            current_weight=current_weight,
            last_reinforced=reference_time,
            stability=stability,
            now=now,
        )

    async def resurrect_memory(
        self, node_id: str, reinforcement_weight: float = 0.8
    ) -> None:
        """
        Resurrect an archived memory when new evidence reinforces it.

        Sets status back to 'active', boosts salience, and increases stability.
        """
        node = await self._neo4j.get_node(node_id)
        if node is None:
            logger.warning(f"Cannot resurrect node {node_id}: not found")
            return

        new_stability = resurrection_stability(
            current_stability=node.stability,
            boost_multiplier=self._settings.decay_resurrection_boost,
        )

        await self._neo4j.update_node(
            node_id,
            {
                "status": "active",
                "salience": min(1.0, node.salience + reinforcement_weight * 0.3),
                "stability": new_stability,
                "last_accessed": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
        logger.info(
            f"Resurrected memory {node_id}: "
            f"new_stability={new_stability:.2f}, salience boosted"
        )
