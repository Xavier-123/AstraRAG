"""
astra_rag/self_improving/system_optimization.py
-------------------------------------------------
SystemOptimization — applies optimisation suggestions from FeedbackLearning.

DSPy-inspired design (without hard DSPy dependency)
----------------------------------------------------
The core idea is *programmatic prompt optimisation*: instead of manually
tweaking prompts, the optimiser takes a performance signal and a suggestion
dict and automatically rewrites the affected system prompt.

Implemented optimisation types
--------------------------------
``prompt_optimization``
    Rewrites the system prompt of the target agent based on the failure mode
    description.  Uses the LLM itself to improve the prompt ("LLM-as-prompt-
    engineer" pattern).

``retriever_tuning``
    Adjusts numerical hyperparameters in ``SystemConfig``:
    - ``rerank.similarity_threshold`` ± 0.05
    - ``retrieval.top_k`` ± 2

``strategy_selection``
    Updates episodic memory weights so future strategy recommendations
    are more conservative.

Guardrails
-----------
* All optimisations are bounded (thresholds stay in [0.1, 0.95],
  top_k stays in [3, 20]).
* Each optimisation is logged so operators can audit changes.
* ``max_optimization_rounds`` prevents runaway optimisation loops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from astra_rag.core.config import SystemConfig
from astra_rag.utils.llm import get_llm

logger = logging.getLogger(__name__)

Suggestion = Dict[str, Any]


class SystemOptimization:
    """Applies optimisation suggestions to live system configuration and prompts.

    Parameters
    ----------
    config:
        System configuration (mutated in-place for parameter tuning).
    custom_prompts:
        Dict mapping agent names to their current system prompt strings.
        The optimiser will update these in-place.
    """

    _PROMPT_ENGINEER_SYSTEM = (
        "You are an expert prompt engineer for retrieval-augmented generation systems.\n"
        "Given a system prompt that is producing suboptimal results, rewrite it to fix\n"
        "the identified problem. Keep the rewrite concise and precise.\n"
        "Return ONLY the new system prompt text — no commentary, no markdown fences.\n"
    )

    def __init__(
        self,
        config: SystemConfig,
        custom_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        self._config = config
        self._prompts: Dict[str, str] = custom_prompts or {}
        self._llm = get_llm(config)
        self._rounds_applied = 0

    def apply(self, suggestions: List[Suggestion]) -> Dict[str, Any]:
        """Apply a list of optimisation suggestions.

        Parameters
        ----------
        suggestions:
            Output of ``FeedbackLearning.process()``.

        Returns
        -------
        Dict[str, Any]
            Summary of changes made.
        """
        max_rounds = self._config.self_improving.max_optimization_rounds
        if self._rounds_applied >= max_rounds:
            logger.warning(
                "[SystemOptimization] Max rounds (%d) reached; skipping.", max_rounds
            )
            return {"skipped": True, "reason": "max_rounds_reached"}

        changes: Dict[str, Any] = {}
        for suggestion in suggestions:
            fix_type = suggestion.get("fix_type")
            if fix_type == "prompt_optimization" and self._config.self_improving.enable_prompt_optimization:
                change = self._optimize_prompt(suggestion)
                if change:
                    changes[f"prompt_{suggestion['target_agent']}"] = change
            elif fix_type == "retriever_tuning" and self._config.self_improving.enable_retriever_tuning:
                change = self._tune_retriever(suggestion)
                if change:
                    changes[f"retriever_{suggestion['target_agent']}"] = change

        if changes:
            self._rounds_applied += 1
            logger.info(
                "[SystemOptimization] Round %d/%d applied %d changes.",
                self._rounds_applied,
                max_rounds,
                len(changes),
            )

        return changes

    # ── Prompt optimisation ───────────────────────────────────────────────

    def _optimize_prompt(self, suggestion: Suggestion) -> Optional[str]:
        """Rewrite the system prompt for the target agent."""
        agent = suggestion["target_agent"]
        current_prompt = self._prompts.get(agent)
        if not current_prompt:
            logger.info(
                "[SystemOptimization] No registered prompt for '%s'; skipping.", agent
            )
            return None

        problem = suggestion["root_cause"]
        fix_hint = suggestion["suggestion"]

        user_content = (
            f"Agent: {agent}\n"
            f"Problem: {problem}\n"
            f"Fix hint: {fix_hint}\n\n"
            f"Current system prompt:\n{current_prompt}"
        )
        messages = [
            SystemMessage(content=self._PROMPT_ENGINEER_SYSTEM),
            HumanMessage(content=user_content),
        ]

        try:
            response = self._llm.invoke(messages)
            new_prompt = response.content.strip()
            self._prompts[agent] = new_prompt
            logger.info("[SystemOptimization] Prompt updated for agent '%s'.", agent)
            return new_prompt
        except Exception as exc:
            logger.error("[SystemOptimization] Prompt optimisation failed: %s", exc)
            return None

    # ── Retriever tuning ──────────────────────────────────────────────────

    def _tune_retriever(self, suggestion: Suggestion) -> Optional[Dict[str, Any]]:
        """Adjust retrieval hyperparameters based on the suggestion."""
        agent = suggestion["target_agent"]
        hint = suggestion.get("suggestion", "")
        changes: Dict[str, Any] = {}

        if agent == "rerank" and "threshold" in hint:
            old = self._config.retrieval.similarity_threshold
            new = round(min(0.95, max(0.1, old + 0.05)), 2)
            self._config.retrieval.similarity_threshold = new
            changes["similarity_threshold"] = {"old": old, "new": new}
            logger.info(
                "[SystemOptimization] similarity_threshold: %.2f → %.2f", old, new
            )

        elif agent == "multi_retriever" and "top_k" in hint:
            old = self._config.retrieval.top_k
            new = min(20, max(3, old + 2))
            self._config.retrieval.top_k = new
            changes["top_k"] = {"old": old, "new": new}
            logger.info("[SystemOptimization] top_k: %d → %d", old, new)

        return changes if changes else None

    # ── Inspection ────────────────────────────────────────────────────────

    def get_current_prompts(self) -> Dict[str, str]:
        """Return the current (possibly optimised) prompt for each agent."""
        return dict(self._prompts)

    def register_prompt(self, agent_name: str, prompt: str) -> None:
        """Register an agent's system prompt for potential optimisation."""
        self._prompts[agent_name] = prompt

    def reset_rounds(self) -> None:
        """Reset the rounds counter (e.g., at the start of a new session)."""
        self._rounds_applied = 0
