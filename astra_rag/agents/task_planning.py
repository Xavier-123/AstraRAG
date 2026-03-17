"""
astra_rag/agents/task_planning.py
----------------------------------
TaskPlanningAgent — Pipeline position: node 2 of 8.

Responsibilities
----------------
* Receives the enriched state from ``QueryUnderstandingAgent`` (intent,
  complexity, sub_queries).
* Produces a **structured execution plan** that tells the rest of the
  pipeline:
  - Which steps to run (retrieval, reasoning, verification…)
  - Whether steps can be executed in parallel
  - Which reasoning mode to use (chain_of_thought | multi_hop | direct)
  - Estimated resource budget (retrieval top-k, context size)

The plan is stored in ``state["task_plan"]`` as a plain ``Dict`` so it can
be inspected / overridden by operators without touching agent code.

Plan schema (example)
----------------------
::

    {
        "steps": [
            {"id": 1, "name": "retrieve",  "agent": "multi_retriever", "parallel": False},
            {"id": 2, "name": "rerank",    "agent": "rerank",           "parallel": False},
            {"id": 3, "name": "reason",    "agent": "reasoning",        "parallel": False},
            {"id": 4, "name": "reflect",   "agent": "reflection",       "parallel": False}
        ],
        "parallel_groups": [],
        "reasoning_mode": "chain_of_thought",
        "top_k_override": null,
        "skip_retrieval": false,
        "rationale": "..."
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)


# ── Structured output schema ──────────────────────────────────────────────────


class PlanStep(BaseModel):
    id: int
    name: str
    agent: str
    parallel: bool = False
    description: Optional[str] = None


class TaskPlan(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list)
    parallel_groups: List[List[int]] = Field(default_factory=list)
    reasoning_mode: str = Field(
        default="chain_of_thought",
        description="chain_of_thought | multi_hop | direct",
    )
    top_k_override: Optional[int] = None
    skip_retrieval: bool = False
    rationale: Optional[str] = None


# ── Agent ─────────────────────────────────────────────────────────────────────


class TaskPlanningAgent(BaseAgent):
    """Converts query understanding output into a concrete execution plan.

    Sets ``state["task_plan"]`` with a dict matching the ``TaskPlan`` schema.
    """

    _SYSTEM_PROMPT = (
        "You are a task-planning expert for an enterprise RAG system.\n"
        "Given a query's intent, complexity, and sub-queries, produce an\n"
        "optimal execution plan.\n\n"
        "Available agents/steps:\n"
        "  multi_retriever  – retrieves documents from vector/graph/web sources\n"
        "  rerank           – reranks and filters retrieved documents\n"
        "  context_engineering – deduplicates, compresses, and assembles context\n"
        "  reasoning        – performs step-by-step or multi-hop reasoning\n"
        "  reflection       – verifies the answer and detects hallucinations\n\n"
        "Reasoning modes:\n"
        "  direct          – simple factual queries, no CoT needed\n"
        "  chain_of_thought – medium/analytical queries\n"
        "  multi_hop       – complex queries requiring iterative retrieval\n\n"
        "Rules:\n"
        "  - Skip retrieval (skip_retrieval=true) only for pure conversational queries.\n"
        "  - Use multi_hop reasoning for complex queries with sub_queries.\n"
        "  - Parallel groups list step IDs that can run concurrently.\n"
        "  - Always include reflection as the last step.\n"
    )

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "task_planning"

    @property
    def description(self) -> str:
        return (
            "Creates an ordered execution plan specifying which agents to invoke "
            "and whether steps can be parallelised."
        )

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        intent = state.get("intent", "factual")
        complexity = state.get("complexity", "simple")
        sub_queries = state.get("sub_queries") or []
        rewritten = state.get("rewritten_query") or state.get("query", "")

        user_content = (
            f"Query: {rewritten}\n"
            f"Intent: {intent}\n"
            f"Complexity: {complexity}\n"
            f"Sub-queries: {json.dumps(sub_queries)}\n\n"
            "Produce the execution plan."
        )

        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            plan = self._structured_call(messages)
        except Exception as exc:
            logger.error("[task_planning] LLM call failed: %s", exc)
            plan = self._default_plan(intent, complexity)

        self._log_exit(state)
        return self._update_state(state, {"task_plan": plan.model_dump()})

    # ── Helpers ───────────────────────────────────────────────────────────

    def _structured_call(self, messages: list) -> TaskPlan:
        llm = self._get_llm()
        try:
            return llm.with_structured_output(TaskPlan).invoke(messages)
        except Exception:
            raw = self._call_llm(
                messages
                + [HumanMessage(content="Return ONLY a JSON object matching the TaskPlan schema.")]
            )
            return TaskPlan(**json.loads(raw.content))

    @staticmethod
    def _default_plan(intent: str, complexity: str) -> TaskPlan:
        """Heuristic fallback when the LLM is unavailable."""
        skip = intent == "conversational"
        mode = (
            "direct"
            if complexity == "simple"
            else ("multi_hop" if complexity == "complex" else "chain_of_thought")
        )
        steps: List[Dict[str, Any]] = []
        sid = 1
        if not skip:
            steps += [
                {"id": sid, "name": "retrieve", "agent": "multi_retriever", "parallel": False},
                {"id": sid + 1, "name": "rerank", "agent": "rerank", "parallel": False},
                {
                    "id": sid + 2,
                    "name": "context",
                    "agent": "context_engineering",
                    "parallel": False,
                },
            ]
            sid += 3
        steps.append({"id": sid, "name": "reason", "agent": "reasoning", "parallel": False})
        steps.append(
            {"id": sid + 1, "name": "reflect", "agent": "reflection", "parallel": False}
        )
        return TaskPlan(
            steps=[PlanStep(**s) for s in steps],
            reasoning_mode=mode,
            skip_retrieval=skip,
            rationale="Default heuristic plan (LLM unavailable).",
        )
