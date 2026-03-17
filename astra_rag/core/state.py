"""
astra_rag/core/state.py
-----------------------
Central LangGraph state definition for the entire AstraRAG pipeline.

Every agent node receives a ``GraphState`` dict and returns a (partial) dict
that LangGraph merges back into the shared state.  Using ``TypedDict`` keeps
the state fully type-checked while remaining compatible with LangGraph's
reducer machinery.

State lifecycle (left-to-right pipeline)
-----------------------------------------
query ──► rewritten_query ──► intent / complexity / sub_queries
      ──► task_plan
      ──► retrieval_strategy
      ──► retrieved_documents ──► reranked_documents ──► context
      ──► reasoning_chain ──► answer
      ──► confidence_score / hallucination_detected / reflection_notes
      ──► evaluation_metrics / feedback (self-improving loop)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state threaded through every node in the LangGraph pipeline.

    Fields marked *Optional* start as ``None`` and are populated by the
    relevant agent.  ``messages`` and ``iteration_count`` always have
    sensible defaults.
    """

    # ── Incoming query ──────────────────────────────────────────────────
    query: str
    """Raw user query as received from the application layer."""

    session_id: str
    """Unique identifier for the conversation session (used by memory)."""

    # ── Query understanding ──────────────────────────────────────────────
    rewritten_query: Optional[str]
    """LLM-improved version of the original query for better retrieval."""

    intent: Optional[str]
    """Detected query intent: factual | analytical | creative | conversational."""

    complexity: Optional[str]
    """Query complexity tier: simple | medium | complex."""

    sub_queries: Optional[List[str]]
    """Decomposed sub-questions for multi-hop / complex queries."""

    # ── Planning ─────────────────────────────────────────────────────────
    task_plan: Optional[Dict[str, Any]]
    """Ordered execution plan produced by TaskPlanningAgent.

    Schema example::

        {
            "steps": [{"id": 1, "agent": "retrieval", "parallel": False}],
            "parallel_groups": [[2, 3]],
            "reasoning_mode": "chain_of_thought"
        }
    """

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_strategy: Optional[str]
    """Strategy chosen by RetrievalPlanningAgent:
    vector | graph | web | hybrid | none."""

    retrieved_documents: Optional[List[Dict[str, Any]]]
    """Raw documents returned by MultiRetrieverAgent.

    Each entry: ``{"id": str, "content": str, "source": str, "score": float}``
    """

    reranked_documents: Optional[List[Dict[str, Any]]]
    """Documents after cross-encoder / LLM reranking, sorted by relevance."""

    # ── Context engineering ──────────────────────────────────────────────
    context: Optional[str]
    """Final assembled context string injected into the reasoning prompt."""

    # ── Reasoning & answer ───────────────────────────────────────────────
    reasoning_chain: Optional[List[str]]
    """Step-by-step reasoning trace built during ReasoningAgent execution."""

    answer: Optional[str]
    """Final natural-language answer surfaced to the user."""

    # ── Reflection & quality ─────────────────────────────────────────────
    confidence_score: Optional[float]
    """Answer confidence in [0, 1] estimated by ReflectionAgent."""

    hallucination_detected: Optional[bool]
    """True if ReflectionAgent found claims unsupported by the context."""

    reflection_notes: Optional[str]
    """Free-text notes from ReflectionAgent about answer quality."""

    # ── Self-improving loop ───────────────────────────────────────────────
    evaluation_metrics: Optional[Dict[str, float]]
    """RAGAS-style scores: faithfulness, answer_relevancy,
    context_precision, context_recall."""

    feedback: Optional[Dict[str, Any]]
    """Structured feedback and optimisation suggestions from FeedbackLearning."""

    # ── Conversation history ──────────────────────────────────────────────
    messages: List[BaseMessage]
    """Full conversation history as LangChain message objects."""

    # ── Control flow ─────────────────────────────────────────────────────
    error: Optional[str]
    """Last error message; non-None triggers error-handling edges."""

    iteration_count: int
    """How many reflection/re-generation cycles have occurred this turn."""

    metadata: Optional[Dict[str, Any]]
    """Arbitrary key-value pairs for debugging, tracing, and extensions."""
