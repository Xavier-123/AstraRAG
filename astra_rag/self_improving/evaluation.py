"""
astra_rag/self_improving/evaluation.py
----------------------------------------
EvaluationLayer — RAGAS-style quality metrics using LLM-as-Judge.

Metrics computed
-----------------
``faithfulness``
    Are all factual claims in the answer grounded in the retrieved context?
    Score: fraction of supported claims.

``answer_relevancy``
    How well does the answer address the original question?
    Score: semantic match between question intent and answer.

``context_precision``
    What fraction of the retrieved context is actually relevant to the
    question?  (Precision of retrieval.)

``context_recall``
    Does the retrieved context contain enough information to fully answer
    the question?  (Recall of retrieval.)

Implementation
--------------
Each metric is computed by asking the LLM to score it on [0, 1] with a
brief justification.  The LLM calls are batched where possible.

For production use, replace with the real ``ragas`` library:
``pip install ragas`` and swap ``_llm_judge_metric`` for the RAGAS
``evaluate()`` function.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from astra_rag.core.config import SystemConfig
from astra_rag.utils.llm import get_llm

logger = logging.getLogger(__name__)

MetricScores = Dict[str, float]


class EvaluationLayer:
    """Computes RAGAS-style evaluation metrics for RAG pipeline outputs.

    Parameters
    ----------
    config:
        System configuration.
    """

    _JUDGE_SYSTEM = (
        "You are an expert evaluator for retrieval-augmented generation systems.\n"
        "Score the following metric on a scale of 0.0 to 1.0 and provide a brief justification.\n"
        "Respond with ONLY a JSON object: {\"score\": <float>, \"justification\": <str>}\n"
    )

    _METRIC_PROMPTS = {
        "faithfulness": (
            "Metric: FAITHFULNESS\n"
            "Definition: Are all factual claims in the answer grounded in the provided context?\n"
            "Score 1.0 = every claim is supported by the context.\n"
            "Score 0.0 = answer contains claims not found in context (hallucinations).\n\n"
            "Context:\n{context}\n\nAnswer:\n{answer}"
        ),
        "answer_relevancy": (
            "Metric: ANSWER RELEVANCY\n"
            "Definition: How directly and completely does the answer address the question?\n"
            "Score 1.0 = answer directly, completely answers the question.\n"
            "Score 0.0 = answer is off-topic or misses the point entirely.\n\n"
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
        "context_precision": (
            "Metric: CONTEXT PRECISION\n"
            "Definition: What fraction of the retrieved context is actually relevant to the question?\n"
            "Score 1.0 = all retrieved context is relevant.\n"
            "Score 0.0 = retrieved context is entirely irrelevant.\n\n"
            "Question:\n{question}\n\nContext:\n{context}"
        ),
        "context_recall": (
            "Metric: CONTEXT RECALL\n"
            "Definition: Does the retrieved context contain all information needed to answer the question?\n"
            "Score 1.0 = context fully covers everything needed.\n"
            "Score 0.0 = context is missing critical information.\n\n"
            "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"
        ),
    }

    def __init__(self, config: SystemConfig) -> None:
        self._llm = get_llm(config)
        self._config = config

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        metrics: Optional[List[str]] = None,
    ) -> MetricScores:
        """Evaluate a single RAG response.

        Parameters
        ----------
        question:
            The original user question.
        answer:
            The generated answer to evaluate.
        context:
            The assembled context used for generation.
        metrics:
            Subset of metrics to compute.  Defaults to all four.

        Returns
        -------
        Dict[str, float]
            ``{metric_name: score}`` dict.
        """
        metrics = metrics or list(self._METRIC_PROMPTS.keys())
        scores: MetricScores = {}

        for metric in metrics:
            if metric not in self._METRIC_PROMPTS:
                logger.warning("[Evaluation] Unknown metric '%s'; skipping.", metric)
                continue
            score = self._compute_metric(metric, question, answer, context)
            scores[metric] = score

        scores["composite"] = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
        logger.info("[Evaluation] Scores: %s", scores)
        return scores

    def evaluate_batch(
        self,
        samples: List[Dict[str, str]],
        metrics: Optional[List[str]] = None,
    ) -> List[MetricScores]:
        """Evaluate multiple RAG responses.

        Parameters
        ----------
        samples:
            List of ``{"question": ..., "answer": ..., "context": ...}`` dicts.
        metrics:
            Metrics to compute for each sample.
        """
        results = []
        for sample in samples:
            result = self.evaluate(
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                context=sample.get("context", ""),
                metrics=metrics,
            )
            results.append(result)
        return results

    # ── Internal ──────────────────────────────────────────────────────────

    def _compute_metric(
        self, metric: str, question: str, answer: str, context: str
    ) -> float:
        template = self._METRIC_PROMPTS[metric]
        # Truncate context to avoid token overflow
        ctx_snippet = context[:2000] if context else "(no context)"
        user_content = template.format(
            question=question[:500],
            answer=answer[:1000],
            context=ctx_snippet,
        )
        messages = [
            SystemMessage(content=self._JUDGE_SYSTEM),
            HumanMessage(content=user_content),
        ]
        try:
            response = self._llm.invoke(messages)
            data = json.loads(response.content)
            return float(data["score"])
        except Exception as exc:
            logger.warning(
                "[Evaluation] Failed to compute metric '%s': %s; defaulting to 0.5.",
                metric,
                exc,
            )
            return 0.5  # neutral fallback
