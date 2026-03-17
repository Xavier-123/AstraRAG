"""
astra_rag/agents/retrieval_planning.py
----------------------------------------
RetrievalPlanningAgent — Pipeline position: node 3 of 8.

Responsibilities
----------------
Select the optimal retrieval strategy for the current query so that
``MultiRetrieverAgent`` knows which backends to activate.

Supported strategies
--------------------
``vector``
    Dense similarity search over the knowledge vector store.
    Best for: semantic questions, document lookup.

``graph``
    Traversal of a knowledge graph (entities + relationships).
    Best for: entity-heavy queries, relationship questions, multi-hop facts.

``web``
    Live internet search.
    Best for: real-time events, news, recent developments.

``hybrid``
    Combines vector + graph (and optionally web) in parallel.
    Best for: complex analytical queries mixing facts and relationships.

``none``
    Skip retrieval entirely.
    Best for: purely conversational or creative intents.

Selection heuristics (applied when LLM is unavailable)
-------------------------------------------------------
* ``conversational`` intent               → ``none``
* ``creative`` intent                     → ``none``
* query contains temporal keywords        → ``web``
* query contains entity/relationship cues → ``graph``
* ``complex`` complexity                  → ``hybrid``
* default                                 → ``vector``
"""

from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)

_TEMPORAL_PATTERNS = re.compile(
    r"\b(today|yesterday|this week|this month|this year|latest|recent|"
    r"current|news|now|2024|2025)\b",
    re.IGNORECASE,
)
_ENTITY_PATTERNS = re.compile(
    r"\b(who is|what is the relationship|how are .+ related|compare|"
    r"difference between|connect|link)\b",
    re.IGNORECASE,
)


# ── Structured output ─────────────────────────────────────────────────────────


class RetrievalDecision(BaseModel):
    strategy: str = Field(description="One of: vector, graph, web, hybrid, none")
    rationale: str = Field(description="One-sentence justification")
    secondary_strategy: str = Field(
        default="",
        description="Optional fallback if primary yields no results",
    )
    sub_query_strategies: List[str] = Field(
        default_factory=list,
        description="Per sub-query strategy overrides (same ordering as sub_queries)",
    )


# ── Agent ─────────────────────────────────────────────────────────────────────


class RetrievalPlanningAgent(BaseAgent):
    """Chooses the retrieval strategy and records it in ``state["retrieval_strategy"]``."""

    _SYSTEM_PROMPT = (
        "You are a retrieval strategy expert for an enterprise RAG system.\n"
        "Choose the best retrieval strategy based on the query characteristics.\n\n"
        "Strategies:\n"
        "  vector  – semantic vector search (facts, definitions, docs)\n"
        "  graph   – knowledge graph traversal (entities, relationships)\n"
        "  web     – live internet search (real-time, news)\n"
        "  hybrid  – vector + graph (complex analytical queries)\n"
        "  none    – no retrieval (conversational, creative)\n\n"
        "Consider: intent, complexity, named entities, temporal references.\n"
    )

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "retrieval_planning"

    @property
    def description(self) -> str:
        return "Selects the optimal retrieval strategy (vector/graph/web/hybrid/none)."

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        task_plan = state.get("task_plan") or {}
        if task_plan.get("skip_retrieval"):
            logger.info("[retrieval_planning] Task plan says skip retrieval → none")
            return self._update_state(state, {"retrieval_strategy": "none"})

        intent = state.get("intent", "factual")
        complexity = state.get("complexity", "simple")
        query = state.get("rewritten_query") or state.get("query", "")
        sub_queries = state.get("sub_queries") or []

        user_content = (
            f"Query: {query}\n"
            f"Intent: {intent}\n"
            f"Complexity: {complexity}\n"
            f"Sub-queries: {sub_queries}\n\n"
            "Choose the retrieval strategy."
        )

        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            llm = self._get_llm()
            decision: RetrievalDecision = llm.with_structured_output(RetrievalDecision).invoke(
                messages
            )
        except Exception as exc:
            logger.warning("[retrieval_planning] LLM failed (%s), using heuristic.", exc)
            decision = self._heuristic_decision(intent, complexity, query)

        logger.info("[retrieval_planning] Strategy selected: %s", decision.strategy)
        self._log_exit(state)
        return self._update_state(state, {"retrieval_strategy": decision.strategy})

    # ── Heuristic fallback ────────────────────────────────────────────────

    @staticmethod
    def _heuristic_decision(
        intent: str, complexity: str, query: str
    ) -> RetrievalDecision:
        if intent in ("conversational", "creative"):
            return RetrievalDecision(strategy="none", rationale="No retrieval needed.")
        if _TEMPORAL_PATTERNS.search(query):
            return RetrievalDecision(strategy="web", rationale="Temporal reference detected.")
        if _ENTITY_PATTERNS.search(query):
            return RetrievalDecision(strategy="graph", rationale="Entity/relationship query.")
        if complexity == "complex":
            return RetrievalDecision(strategy="hybrid", rationale="Complex query benefits from hybrid.")
        return RetrievalDecision(strategy="vector", rationale="Default semantic search.")
