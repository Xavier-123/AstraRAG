"""
astra_rag/workflow/graph.py
----------------------------
LangGraph pipeline definition and ``AstraRAGSystem`` entry-point class.

Pipeline overview (节点 = node，边 = edge)
==========================================

                    ┌─────────────────────┐
                    │  query_understanding │  Node 1 – 分析查询意图/复杂度
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │    task_planning    │  Node 2 – 制定执行计划
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ retrieval_planning  │  Node 3 – 选择检索策略
                    └────────┬────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │  strategy == "none"?             │  Conditional edge
              │  yes → skip to reasoning         │
              │  no  → multi_retriever           │
              └──────────────┬──────────────────┘
                             │ (retrieval path)
                    ┌────────▼────────────┐
                    │   multi_retriever   │  Node 4 – 并行检索
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │       rerank        │  Node 5 – 重排序/过滤
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ context_engineering │  Node 6 – 上下文工程
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │      reasoning      │  Node 7 – 推理/生成答案
                    └────────┬────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │  multi-hop retrieval needed?     │  Conditional edge
              │  yes → loop back to retrieval   │
              │  no  → reflection               │
              └──────────────┬──────────────────┘
                             │
                    ┌────────▼────────────┐
                    │      reflection     │  Node 8 – 质量检查/幻觉检测
                    └────────┬────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │  quality ok?                     │  Conditional edge
              │  yes → END                      │
              │  no  → reasoning (re-generate)  │
              └──────────────┴──────────────────┘

Self-improving loop
-------------------
After each complete run, ``AstraRAGSystem.run()`` optionally triggers the
EvaluationLayer → FeedbackLearning → SystemOptimization cycle.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from astra_rag.agents.context_engineering import ContextEngineeringAgent
from astra_rag.agents.multi_retriever import MultiRetrieverAgent
from astra_rag.agents.query_understanding import QueryUnderstandingAgent
from astra_rag.agents.reasoning import ReasoningAgent
from astra_rag.agents.reflection import ReflectionAgent
from astra_rag.agents.rerank import RerankAgent
from astra_rag.agents.retrieval_planning import RetrievalPlanningAgent
from astra_rag.agents.task_planning import TaskPlanningAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState
from astra_rag.memory.conversation_memory import ConversationMemory
from astra_rag.memory.episodic_memory import EpisodicMemory
from astra_rag.self_improving.evaluation import EvaluationLayer
from astra_rag.self_improving.feedback_learning import FeedbackLearning
from astra_rag.self_improving.system_optimization import SystemOptimization

logger = logging.getLogger(__name__)


# ── Routing helpers ────────────────────────────────────────────────────────────


def _route_after_retrieval_planning(state: GraphState) -> str:
    """Skip retrieval entirely for conversational/creative queries."""
    if state.get("retrieval_strategy") == "none":
        return "reasoning"
    return "multi_retriever"


def _route_after_reasoning(state: GraphState) -> str:
    """Loop back to retrieval if the reasoning agent requested more info."""
    # Multi-hop signal: rewritten_query changed AND no final answer yet
    original = state.get("query", "")
    rewritten = state.get("rewritten_query", "")
    answer = state.get("answer")
    # If there is no answer and iteration_count increased, we need more retrieval
    iteration = state.get("iteration_count", 0)
    if not answer and iteration > 0 and rewritten != original:
        return "multi_retriever"
    return "reflection"


def _route_after_reflection(state: GraphState) -> str:
    """Re-generate if quality is insufficient; otherwise finish."""
    answer = state.get("answer")
    confidence = state.get("confidence_score", 1.0)
    iteration = state.get("iteration_count", 0)
    hallucination = state.get("hallucination_detected", False)
    max_iters = 3  # default; overridden by config in build_graph

    if (not answer or hallucination or confidence < 0.6) and iteration < max_iters:
        return "reasoning"
    return END


# ── Graph builder ──────────────────────────────────────────────────────────────


def build_graph(
    config: SystemConfig,
    knowledge_memory: Optional[Any] = None,
    graph_client: Optional[Any] = None,
    search_client: Optional[Any] = None,
) -> Any:
    """Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    config:
        System configuration.
    knowledge_memory:
        Optional ``KnowledgeMemory`` instance for vector retrieval.
    graph_client:
        Optional graph DB client for ``GraphRetriever``.
    search_client:
        Optional web search client for ``WebRetriever``.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready for ``invoke``/``ainvoke``.
    """
    # ── Instantiate all agents ─────────────────────────────────────────
    query_understanding = QueryUnderstandingAgent(config)
    task_planning = TaskPlanningAgent(config)
    retrieval_planning = RetrievalPlanningAgent(config)
    multi_retriever = MultiRetrieverAgent(
        config,
        knowledge_memory=knowledge_memory,
        graph_client=graph_client,
        search_client=search_client,
    )
    rerank = RerankAgent(config)
    context_engineering = ContextEngineeringAgent(config)
    reasoning = ReasoningAgent(config)
    reflection = ReflectionAgent(config)

    # ── Closure over max_iters for the reflection router ──────────────
    max_iters = config.max_reflection_iterations

    def _route_reflection(state: GraphState) -> str:
        answer = state.get("answer")
        confidence = state.get("confidence_score", 1.0)
        iteration = state.get("iteration_count", 0)
        hallucination = state.get("hallucination_detected", False)
        if (not answer or hallucination or confidence < 0.6) and iteration < max_iters:
            return "reasoning"
        return END

    # ── Build StateGraph ───────────────────────────────────────────────
    graph = StateGraph(GraphState)

    # Add nodes (each agent's run() method is the node function)
    graph.add_node("query_understanding", query_understanding.run)
    graph.add_node("task_planning", task_planning.run)
    graph.add_node("retrieval_planning", retrieval_planning.run)
    graph.add_node("multi_retriever", multi_retriever.run)
    graph.add_node("rerank", rerank.run)
    graph.add_node("context_engineering", context_engineering.run)
    graph.add_node("reasoning", reasoning.run)
    graph.add_node("reflection", reflection.run)

    # Linear edges (no branching)
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "task_planning")
    graph.add_edge("task_planning", "retrieval_planning")

    # Conditional edge: skip retrieval for none strategy
    graph.add_conditional_edges(
        "retrieval_planning",
        _route_after_retrieval_planning,
        {
            "reasoning": "reasoning",
            "multi_retriever": "multi_retriever",
        },
    )

    # Retrieval sub-pipeline
    graph.add_edge("multi_retriever", "rerank")
    graph.add_edge("rerank", "context_engineering")
    graph.add_edge("context_engineering", "reasoning")

    # Conditional edge: multi-hop or reflection
    graph.add_conditional_edges(
        "reasoning",
        _route_after_reasoning,
        {
            "multi_retriever": "multi_retriever",
            "reflection": "reflection",
        },
    )

    # Conditional edge: re-generate or finish
    graph.add_conditional_edges(
        "reflection",
        _route_reflection,
        {
            "reasoning": "reasoning",
            END: END,
        },
    )

    return graph.compile()


# ── AstraRAGSystem ─────────────────────────────────────────────────────────────


class AstraRAGSystem:
    """High-level façade wrapping the compiled LangGraph pipeline.

    Provides synchronous ``run()`` and asynchronous ``arun()`` entry points,
    conversation memory management, and the self-improving feedback loop.

    Parameters
    ----------
    config:
        System configuration.  Defaults to ``SystemConfig()`` (reads env vars).
    knowledge_memory:
        Optional pre-built ``KnowledgeMemory`` for vector retrieval.
    graph_client:
        Optional graph DB client.
    search_client:
        Optional web search client.
    enable_self_improving:
        Whether to run the evaluation/feedback/optimisation cycle.

    Example::

        system = AstraRAGSystem()
        result = system.run("What is the capital of France?")
        print(result["answer"])  # Paris
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        knowledge_memory: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        search_client: Optional[Any] = None,
        enable_self_improving: bool = True,
    ) -> None:
        self.config = config or SystemConfig()
        self._graph = build_graph(
            self.config,
            knowledge_memory=knowledge_memory,
            graph_client=graph_client,
            search_client=search_client,
        )
        self._conv_memory = ConversationMemory(self.config)
        self._episodic = EpisodicMemory()
        self._enable_si = enable_self_improving

        if enable_self_improving:
            self._evaluator = EvaluationLayer(self.config)
            self._feedback = FeedbackLearning(self.config, self._episodic)
            self._optimizer = SystemOptimization(self.config)

        logger.info("[AstraRAGSystem] Initialised (model=%s)", self.config.llm.model)

    # ── Synchronous API ────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """Run the full RAG pipeline synchronously.

        Parameters
        ----------
        query:
            User's natural-language question.
        session_id:
            Conversation session ID (generated if not provided).
        metadata:
            Optional key-value pairs forwarded through the pipeline.

        Returns
        -------
        GraphState
            Final pipeline state containing ``answer``, ``confidence_score``,
            ``reasoning_chain``, ``evaluation_metrics``, etc.
        """
        session_id = session_id or str(uuid.uuid4())
        history = self._conv_memory.format_for_prompt(session_id, max_turns=5)

        initial_state: GraphState = {
            "query": query,
            "session_id": session_id,
            "messages": history,
            "iteration_count": 0,
            "metadata": metadata or {},
            # Optional fields initialised to None / defaults
            "rewritten_query": None,
            "intent": None,
            "complexity": None,
            "sub_queries": None,
            "task_plan": None,
            "retrieval_strategy": None,
            "retrieved_documents": None,
            "reranked_documents": None,
            "context": None,
            "reasoning_chain": None,
            "answer": None,
            "confidence_score": None,
            "hallucination_detected": None,
            "reflection_notes": None,
            "evaluation_metrics": None,
            "feedback": None,
            "error": None,
        }

        logger.info("[AstraRAGSystem] Running query (session=%s): %s", session_id, query[:80])
        final_state: GraphState = self._graph.invoke(initial_state)  # type: ignore[assignment]

        # Persist conversation turn
        self._conv_memory.add_message(session_id, HumanMessage(content=query))
        answer = final_state.get("answer") or ""
        if answer:
            self._conv_memory.add_message(session_id, AIMessage(content=answer))

        # Self-improving loop (async-safe: runs synchronously here)
        if self._enable_si and answer:
            final_state = self._run_self_improving(final_state)

        return final_state

    # ── Async API ──────────────────────────────────────────────────────────

    async def arun(
        self,
        query: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """Async version of :meth:`run`.

        Uses the event loop to call LangGraph's ``ainvoke`` so all I/O
        (LLM calls, retrieval) can be awaited concurrently.
        """
        session_id = session_id or str(uuid.uuid4())
        history = self._conv_memory.format_for_prompt(session_id, max_turns=5)

        initial_state: GraphState = {
            "query": query,
            "session_id": session_id,
            "messages": history,
            "iteration_count": 0,
            "metadata": metadata or {},
            "rewritten_query": None,
            "intent": None,
            "complexity": None,
            "sub_queries": None,
            "task_plan": None,
            "retrieval_strategy": None,
            "retrieved_documents": None,
            "reranked_documents": None,
            "context": None,
            "reasoning_chain": None,
            "answer": None,
            "confidence_score": None,
            "hallucination_detected": None,
            "reflection_notes": None,
            "evaluation_metrics": None,
            "feedback": None,
            "error": None,
        }

        logger.info("[AstraRAGSystem] Async running (session=%s): %s", session_id, query[:80])
        final_state: GraphState = await self._graph.ainvoke(initial_state)  # type: ignore[assignment]

        self._conv_memory.add_message(session_id, HumanMessage(content=query))
        answer = final_state.get("answer") or ""
        if answer:
            self._conv_memory.add_message(session_id, AIMessage(content=answer))

        if self._enable_si and answer:
            final_state = self._run_self_improving(final_state)

        return final_state

    # ── Self-improving ─────────────────────────────────────────────────────

    def _run_self_improving(self, state: GraphState) -> GraphState:
        """Evaluate the response and trigger optimisation if needed."""
        try:
            metrics = self._evaluator.evaluate(
                question=state.get("query", ""),
                answer=state.get("answer", ""),
                context=state.get("context") or "",
            )
            suggestions = self._feedback.process(dict(state), metrics)
            if suggestions:
                changes = self._optimizer.apply(suggestions)
                logger.info("[AstraRAGSystem] Optimisation changes: %s", changes)
            state = {**state, "evaluation_metrics": metrics, "feedback": {"suggestions": suggestions}}  # type: ignore[assignment]
        except Exception as exc:
            logger.warning("[AstraRAGSystem] Self-improving cycle failed: %s", exc)
        return state

    # ── Convenience methods ────────────────────────────────────────────────

    def clear_memory(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._conv_memory.clear(session_id)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return aggregate system performance metrics."""
        if not self._enable_si:
            return {"self_improving": "disabled"}
        return self._feedback.get_performance_summary()
