"""
MNEMOS — Persistent semantic memory graph service for multi-agent AI systems.

Architecture:
    - FastAPI service layer
    - Neo4j hybrid memory graph (episodic / semantic / procedural nodes)
    - Qdrant vector similarity search over memory embeddings
    - Temporal decay engine (Ebbinghaus forgetting curve)
    - ContradictionResolver (LLM micro-agent adjudication)
    - Python SDK for Hermes/Kavi Claw integration

Author: Onur Kavi
"""

__version__ = "0.1.0"
__author__ = "Onur Kavi"
__license__ = "MIT"

from mnemos.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
