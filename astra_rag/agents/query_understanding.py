"""
astra_rag/agents/query_understanding.py
----------------------------------------
QueryUnderstandingAgent — Pipeline position: node 1 of 8.

Responsibilities
----------------
1. **Query rewriting** – rephrase the user's raw query for clarity and
   better retrieval recall (fixing typos, expanding abbreviations, etc.).
2. **Intent detection** – classify the query into one of:
   ``factual`` | ``analytical`` | ``creative`` | ``conversational``.
3. **Complexity estimation** – rate the query as ``simple`` | ``medium`` |
   ``complex`` so downstream planners can allocate resources accordingly.
4. **Query decomposition** – break complex multi-hop queries into a list of
   self-contained sub-queries that can be retrieved independently.

Implementation details
-----------------------
* Uses a **structured LLM call** via Pydantic v2 ``with_structured_output``
  to guarantee parseable JSON output in a single round-trip.
* Falls back to the original query string if the LLM call fails, so the
  pipeline never stalls at this node.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)


# ── Structured output schema ──────────────────────────────────────────────────


class QueryAnalysis(BaseModel):
    """Structured output produced by the QueryUnderstandingAgent."""

    rewritten_query: str = Field(
        description="Clearer version of the query; identical to original if already clear."
    )
    intent: str = Field(
        description="One of: factual, analytical, creative, conversational."
    )
    complexity: str = Field(
        description="One of: simple, medium, complex."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions for complex queries; empty for simple ones.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the classification decisions.",
    )


# ── Agent ─────────────────────────────────────────────────────────────────────


class QueryUnderstandingAgent(BaseAgent):
    """Analyses the raw user query and enriches the state with structured
    understanding before any retrieval occurs.

    This agent sets the following state keys:
    * ``rewritten_query``
    * ``intent``
    * ``complexity``
    * ``sub_queries``
    """

    _SYSTEM_PROMPT = (
        "You are an expert query analyst for an enterprise RAG system.\n"
        "Your job is to deeply understand user queries and prepare them for\n"
        "optimal information retrieval.\n\n"
        "Guidelines:\n"
        "- rewritten_query: Fix grammar/typos, expand acronyms, add context from history.\n"
        "  Keep the user's original meaning intact.\n"
        "- intent:\n"
        "    factual       → looking for a specific fact or definition\n"
        "    analytical    → requires reasoning over multiple pieces of information\n"
        "    creative      → open-ended generation (write, brainstorm, etc.)\n"
        "    conversational → chit-chat, follow-up, or clarification\n"
        "- complexity:\n"
        "    simple  → single-hop, directly answerable\n"
        "    medium  → needs 2-3 pieces of information\n"
        "    complex → multi-hop, comparison, or requires synthesis\n"
        "- sub_queries: only for complex queries; empty list otherwise.\n"
    )

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "query_understanding"

    @property
    def description(self) -> str:
        return (
            "Rewrites the user query, detects intent and complexity, "
            "and decomposes complex queries into sub-questions."
        )

    def run(self, state: GraphState) -> GraphState:
        """Analyse the query and enrich the state.

        Parameters
        ----------
        state:
            Must contain ``query`` and optionally ``messages`` for context.
        """
        self._log_entry(state)

        query = state.get("query", "")
        if not query:
            logger.warning("[query_understanding] Empty query received.")
            return self._update_state(
                state,
                {
                    "rewritten_query": "",
                    "intent": "conversational",
                    "complexity": "simple",
                    "sub_queries": [],
                    "error": "Empty query",
                },
            )

        # Build conversation context from recent messages
        history_snippet = self._build_history_snippet(state)
        user_content = (
            f"Conversation context:\n{history_snippet}\n\n"
            f"Current query: {query}"
        ).strip()

        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            analysis = self._structured_call(messages)
        except Exception as exc:
            logger.error("[query_understanding] LLM call failed: %s", exc)
            # Graceful fallback
            analysis = QueryAnalysis(
                rewritten_query=query,
                intent="factual",
                complexity="simple",
                sub_queries=[],
            )

        self._log_exit(state)
        return self._update_state(
            state,
            {
                "rewritten_query": analysis.rewritten_query,
                "intent": analysis.intent,
                "complexity": analysis.complexity,
                "sub_queries": analysis.sub_queries,
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _structured_call(self, messages: list) -> QueryAnalysis:
        """Invoke the LLM with structured output, falling back to JSON parsing."""
        llm = self._get_llm()
        try:
            structured_llm = llm.with_structured_output(QueryAnalysis)
            return structured_llm.invoke(messages)
        except Exception:
            # Fallback: ask for JSON explicitly
            raw = self._call_llm(
                messages
                + [
                    HumanMessage(
                        content=(
                            "Return ONLY a JSON object with keys: "
                            "rewritten_query, intent, complexity, sub_queries."
                        )
                    )
                ]
            )
            data = json.loads(raw.content)
            return QueryAnalysis(**data)

    @staticmethod
    def _build_history_snippet(state: GraphState, max_turns: int = 3) -> str:
        """Format the last *max_turns* messages for the prompt."""
        messages = state.get("messages", [])
        recent = messages[-(max_turns * 2) :] if messages else []
        if not recent:
            return "(no prior conversation)"
        lines = []
        for m in recent:
            role = type(m).__name__.replace("Message", "").upper()
            content = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"{role}: {content[:200]}")
        return "\n".join(lines)
