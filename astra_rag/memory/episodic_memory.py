"""
astra_rag/memory/episodic_memory.py
-------------------------------------
EpisodicMemory — stores and learns from system "experiences".

Concept
-------
Inspired by cognitive science's episodic memory model, this module tracks
*which retrieval strategies led to high-quality answers* for different
query types.  Over time the system learns to skip unsuccessful strategies
and prefer ones that historically perform well for a given (intent, complexity)
combination.

Data model
-----------
An *episode* represents a single RAG interaction::

    {
        "session_id": str,
        "query_hash":  str,            # MD5 of the rewritten query
        "intent":      str,
        "complexity":  str,
        "strategy":    str,            # retrieval strategy used
        "confidence":  float,          # final confidence score
        "timestamp":   float,          # Unix timestamp
    }

Storage
--------
In-memory ``deque`` with optional JSON file persistence.
Call ``save(path)`` / ``load(path)`` to persist across restarts.

Strategy recommendation
-----------------------
``recommend_strategy(intent, complexity)``
    Returns the strategy that achieved the highest average confidence for
    this (intent, complexity) pair.  Falls back to ``"vector"`` if no
    history exists.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

Episode = Dict[str, Any]
_MAX_EPISODES = 1000  # Rolling window


class EpisodicMemory:
    """Tracks RAG pipeline experiences and recommends retrieval strategies.

    Parameters
    ----------
    max_episodes:
        Maximum episodes to keep in memory (FIFO eviction).
    persist_path:
        Optional file path for JSON persistence across restarts.
    """

    def __init__(
        self,
        max_episodes: int = _MAX_EPISODES,
        persist_path: Optional[str] = None,
    ) -> None:
        self._episodes: deque[Episode] = deque(maxlen=max_episodes)
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self.load(str(self._persist_path))

    # ── Write ─────────────────────────────────────────────────────────────

    def record(
        self,
        session_id: str,
        query: str,
        intent: str,
        complexity: str,
        strategy: str,
        confidence: float,
    ) -> None:
        """Record the outcome of a RAG interaction.

        Parameters
        ----------
        session_id:
            Conversation session identifier.
        query:
            The (rewritten) query string (hashed before storage).
        intent:
            Detected query intent.
        complexity:
            Query complexity tier.
        strategy:
            Retrieval strategy that was executed.
        confidence:
            Final confidence score from ``ReflectionAgent``.
        """
        episode: Episode = {
            "session_id": session_id,
            "query_hash": md5(query.encode()).hexdigest(),
            "intent": intent,
            "complexity": complexity,
            "strategy": strategy,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        self._episodes.append(episode)
        logger.debug(
            "[EpisodicMemory] Recorded episode: strategy=%s confidence=%.2f",
            strategy,
            confidence,
        )
        if self._persist_path:
            self.save(str(self._persist_path))

    # ── Read ──────────────────────────────────────────────────────────────

    def recommend_strategy(self, intent: str, complexity: str) -> str:
        """Return the retrieval strategy with the highest average confidence
        for the given (intent, complexity) combination.

        Falls back to ``"vector"`` if no relevant history exists.
        """
        scores: Dict[str, List[float]] = defaultdict(list)
        for ep in self._episodes:
            if ep["intent"] == intent and ep["complexity"] == complexity:
                scores[ep["strategy"]].append(ep["confidence"])

        if not scores:
            return "vector"  # safe default

        avg_scores = {s: sum(v) / len(v) for s, v in scores.items()}
        best = max(avg_scores, key=lambda s: avg_scores[s])
        logger.debug(
            "[EpisodicMemory] Recommended strategy='%s' (avg=%.2f) for intent=%s complexity=%s",
            best,
            avg_scores[best],
            intent,
            complexity,
        )
        return best

    def get_performance_trends(self) -> Dict[str, Any]:
        """Return aggregate performance metrics grouped by strategy."""
        scores: Dict[str, List[float]] = defaultdict(list)
        for ep in self._episodes:
            scores[ep["strategy"]].append(ep["confidence"])

        return {
            strategy: {
                "count": len(vals),
                "avg_confidence": round(sum(vals) / len(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
            for strategy, vals in scores.items()
        }

    def get_recent_episodes(self, n: int = 20) -> List[Episode]:
        """Return the *n* most recent episodes."""
        episodes = list(self._episodes)
        return episodes[-n:]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist episodes to a JSON file."""
        try:
            Path(path).write_text(
                json.dumps(list(self._episodes), indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[EpisodicMemory] Failed to save: %s", exc)

    def load(self, path: str) -> None:
        """Load episodes from a JSON file."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._episodes = deque(data, maxlen=self._episodes.maxlen)
            logger.info("[EpisodicMemory] Loaded %d episodes from %s", len(self._episodes), path)
        except Exception as exc:
            logger.warning("[EpisodicMemory] Failed to load: %s", exc)
