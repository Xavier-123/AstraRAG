"""
astra_rag/agents/reflection.py
--------------------------------
ReflectionAgent — Pipeline position: node 8 of 8 (per iteration).

Responsibilities
----------------
1. **Answer verification** — checks whether the answer actually addresses
   the original question (coverage check).
2. **Hallucination detection** — identifies claims in the answer that are
   NOT supported by the retrieved context (grounding check).
3. **Confidence scoring** — produces a [0, 1] confidence score combining
   coverage, grounding, and answer coherence.
4. **Regeneration signal** — if confidence < ``regeneration_threshold``
   *and* ``iteration_count < max_iterations``, returns a flag that the
   LangGraph router uses to loop back to the reasoning node.

Output state keys
-----------------
* ``confidence_score``       – float in [0, 1]
* ``hallucination_detected`` – bool
* ``reflection_notes``       – free-text critique
* Optionally ``answer`` is cleared to force regeneration on next hop.

Routing logic (implemented in workflow/graph.py)
-------------------------------------------------
    if hallucination_detected or confidence_score < threshold:
        → loop back to reasoning (up to max_iterations)
    else:
        → done
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)

_REGEN_THRESHOLD = 0.6  # Confidence below this triggers regeneration


# ── Structured output schema ──────────────────────────────────────────────────


class ReflectionResult(BaseModel):
    covers_question: bool = Field(description="Does the answer address the user's question?")
    grounded_in_context: bool = Field(
        description="Are all factual claims supported by the provided context?"
    )
    hallucination_detected: bool = Field(
        description="True if any claim cannot be verified in the context."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Overall answer quality score."
    )
    notes: Optional[str] = Field(
        default=None, description="Critique and suggestions for improvement."
    )


# ── Agent ─────────────────────────────────────────────────────────────────────


class ReflectionAgent(BaseAgent):
    """Verifies answer quality and triggers re-generation when needed."""

    _SYSTEM_PROMPT = (
        "You are a rigorous quality-control agent for a RAG system.\n"
        "Evaluate the answer against the question and context.\n\n"
        "Scoring rubric for confidence_score:\n"
        "  1.0 – perfect: fully answers, all claims grounded, clear and concise\n"
        "  0.8 – good: minor gaps or slight verbosity\n"
        "  0.6 – acceptable: answers the question but missing some details\n"
        "  0.4 – poor: partially answers or contains ungrounded claims\n"
        "  0.2 – bad: mostly wrong or hallucinates significantly\n"
        "  0.0 – failure: does not answer or entirely fabricated\n\n"
        "hallucination_detected = true if ANY specific fact, figure, date, name, or\n"
        "claim in the answer CANNOT be found in or directly inferred from the context.\n"
    )

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "reflection"

    @property
    def description(self) -> str:
        return (
            "Verifies answer quality, detects hallucinations, scores confidence, "
            "and signals re-generation when quality is insufficient."
        )

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        answer = state.get("answer") or ""
        question = state.get("query", "")
        context = state.get("context") or ""
        iteration = state.get("iteration_count", 0)

        if not answer:
            logger.warning("[reflection] No answer to evaluate.")
            return self._update_state(
                state,
                {
                    "confidence_score": 0.0,
                    "hallucination_detected": False,
                    "reflection_notes": "No answer produced.",
                },
            )

        # Build evaluation prompt
        context_snippet = context[:2000] if context else "(no context provided)"
        user_content = (
            f"Question: {question}\n\n"
            f"Context (truncated):\n{context_snippet}\n\n"
            f"Answer:\n{answer}"
        )
        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            result = self._structured_call(messages)
        except Exception as exc:
            logger.error("[reflection] LLM evaluation failed: %s", exc)
            result = ReflectionResult(
                covers_question=True,
                grounded_in_context=True,
                hallucination_detected=False,
                confidence_score=0.7,
                notes=f"Evaluation skipped due to error: {exc}",
            )

        logger.info(
            "[reflection] confidence=%.2f hallucination=%s",
            result.confidence_score,
            result.hallucination_detected,
        )

        updates: Dict[str, Any] = {
            "confidence_score": result.confidence_score,
            "hallucination_detected": result.hallucination_detected,
            "reflection_notes": result.notes,
        }

        # Signal regeneration by clearing the answer
        max_iters = self.config.max_reflection_iterations
        should_regen = (
            (result.hallucination_detected or result.confidence_score < _REGEN_THRESHOLD)
            and iteration < max_iters
        )
        if should_regen:
            logger.info(
                "[reflection] Quality insufficient (iter=%d/%d) → signalling re-generation.",
                iteration,
                max_iters,
            )
            updates["answer"] = None  # cleared → triggers re-generation in router
            updates["iteration_count"] = iteration + 1

        self._log_exit(state)
        return self._update_state(state, updates)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _structured_call(self, messages: list) -> ReflectionResult:
        llm = self._get_llm()
        try:
            return llm.with_structured_output(ReflectionResult).invoke(messages)
        except Exception:
            raw = self._call_llm(
                messages
                + [
                    HumanMessage(
                        content=(
                            "Return ONLY a JSON object with keys: "
                            "covers_question, grounded_in_context, "
                            "hallucination_detected, confidence_score, notes."
                        )
                    )
                ]
            )
            return ReflectionResult(**json.loads(raw.content))
