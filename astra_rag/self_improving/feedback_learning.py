"""
astra_rag/self_improving/feedback_learning.py
-----------------------------------------------
FeedbackLearning — processes evaluation results and generates actionable
optimisation suggestions.

How it works
------------
After each interaction (or batch of interactions) the FeedbackLearning
module:

1. **Trend analysis** — accumulates evaluation scores and computes rolling
   averages per metric.
2. **Weakness identification** — flags metrics whose average score falls
   below ``config.self_improving.optimization_threshold``.
3. **Root-cause mapping** — maps low metrics to likely root causes and
   candidate fixes:
   - low ``faithfulness``     → hallucination in reasoning; tighten system prompt
   - low ``context_precision`` → retriever returning irrelevant docs; raise threshold
   - low ``context_recall``   → retriever missing docs; increase top-k
   - low ``answer_relevancy``  → query understanding or reasoning prompt issue
4. **Episodic memory update** — writes the interaction outcome to
   ``EpisodicMemory`` so strategy recommendations improve over time.
5. **Suggestion generation** — returns a structured dict of optimisation
   hints consumed by ``SystemOptimization``.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from astra_rag.core.config import SystemConfig
from astra_rag.memory.episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)

Suggestion = Dict[str, Any]
MetricScores = Dict[str, float]

_ROOT_CAUSE_MAP: Dict[str, Dict[str, Any]] = {
    "faithfulness": {
        "root_cause": "Reasoning agent is generating unsupported claims.",
        "fix_type": "prompt_optimization",
        "target_agent": "reasoning",
        "suggestion": "Add explicit instruction: 'Only state facts present in the context.'",
    },
    "context_precision": {
        "root_cause": "Retriever is returning irrelevant documents.",
        "fix_type": "retriever_tuning",
        "target_agent": "rerank",
        "suggestion": "Increase rerank similarity_threshold by 0.05.",
    },
    "context_recall": {
        "root_cause": "Retriever is missing relevant documents.",
        "fix_type": "retriever_tuning",
        "target_agent": "multi_retriever",
        "suggestion": "Increase retrieval top_k by 2.",
    },
    "answer_relevancy": {
        "root_cause": "Answer is not directly addressing the question.",
        "fix_type": "prompt_optimization",
        "target_agent": "reasoning",
        "suggestion": "Prepend 'Directly answer: {question}' to the reasoning prompt.",
    },
}


class FeedbackLearning:
    """Processes evaluation scores and generates optimisation suggestions.

    Parameters
    ----------
    config:
        System configuration.
    episodic_memory:
        ``EpisodicMemory`` instance for recording interaction outcomes.
    history_window:
        Number of recent evaluations to consider for trend analysis.
    """

    def __init__(
        self,
        config: SystemConfig,
        episodic_memory: Optional[EpisodicMemory] = None,
        history_window: int = 50,
    ) -> None:
        self._config = config
        self._episodic = episodic_memory or EpisodicMemory()
        self._threshold = config.self_improving.optimization_threshold
        self._history: deque[MetricScores] = deque(maxlen=history_window)
        self._interaction_count = 0

    def process(
        self,
        state: Dict[str, Any],
        evaluation_metrics: MetricScores,
    ) -> List[Suggestion]:
        """Process one interaction's evaluation results.

        Parameters
        ----------
        state:
            The final pipeline state for this interaction.
        evaluation_metrics:
            Metric scores from ``EvaluationLayer``.

        Returns
        -------
        List[Suggestion]
            Ordered list of optimisation suggestions (most impactful first).
        """
        self._interaction_count += 1
        self._history.append(evaluation_metrics)

        # Record in episodic memory
        self._record_episode(state, evaluation_metrics)

        # Only generate suggestions at evaluation_interval
        if self._interaction_count % self._config.self_improving.evaluation_interval != 0:
            return []

        return self._generate_suggestions()

    def _record_episode(self, state: Dict[str, Any], metrics: MetricScores) -> None:
        self._episodic.record(
            session_id=state.get("session_id", "unknown"),
            query=state.get("rewritten_query") or state.get("query", ""),
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "simple"),
            strategy=state.get("retrieval_strategy", "vector"),
            confidence=float(metrics.get("composite", 0.5)),
        )

    def _generate_suggestions(self) -> List[Suggestion]:
        """Analyse rolling averages and produce ranked suggestions."""
        avg_scores = self._rolling_averages()
        suggestions: List[Suggestion] = []

        for metric, avg in avg_scores.items():
            if metric == "composite":
                continue
            if avg < self._threshold and metric in _ROOT_CAUSE_MAP:
                info = _ROOT_CAUSE_MAP[metric]
                suggestions.append(
                    {
                        "metric": metric,
                        "avg_score": round(avg, 3),
                        "threshold": self._threshold,
                        "root_cause": info["root_cause"],
                        "fix_type": info["fix_type"],
                        "target_agent": info["target_agent"],
                        "suggestion": info["suggestion"],
                        "priority": round(self._threshold - avg, 3),  # gap = priority
                    }
                )

        # Sort by priority (biggest gap first)
        suggestions.sort(key=lambda s: s["priority"], reverse=True)

        if suggestions:
            logger.info(
                "[FeedbackLearning] Generated %d suggestions at interaction %d",
                len(suggestions),
                self._interaction_count,
            )
            for s in suggestions:
                logger.info(
                    "  [%s] avg=%.2f → %s", s["metric"], s["avg_score"], s["suggestion"]
                )

        return suggestions

    def _rolling_averages(self) -> Dict[str, float]:
        if not self._history:
            return {}
        totals: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for scores in self._history:
            for metric, val in scores.items():
                totals[metric] += val
                counts[metric] += 1
        return {m: totals[m] / counts[m] for m in totals}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return aggregate performance statistics for monitoring dashboards."""
        avg = self._rolling_averages()
        return {
            "interaction_count": self._interaction_count,
            "rolling_averages": {k: round(v, 3) for k, v in avg.items()},
            "episodic_trends": self._episodic.get_performance_trends(),
        }
