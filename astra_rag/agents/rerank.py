"""
astra_rag/agents/rerank.py
---------------------------
RerankAgent — Pipeline position: node 5 of 8.

Responsibilities
----------------
* Accepts the raw ``retrieved_documents`` list and the current query.
* Scores each document for relevance to the query using an LLM-as-judge
  approach (or a cross-encoder when one is available).
* Filters out documents whose relevance score falls below the configured
  ``similarity_threshold``.
* Returns ``reranked_documents`` sorted by descending relevance.

Reranking modes
---------------
``llm``  (default)
    Sends a batch of up to ``batch_size`` documents to the LLM with a
    scoring prompt and parses numeric scores.  Slower but highly accurate.

``cross_encoder``
    Uses a local ``sentence-transformers`` cross-encoder model for fast,
    offline reranking.  Falls back to ``llm`` if the model fails to load.

The mode is selected automatically: cross-encoder is tried first (zero API
cost); LLM scoring is used when no cross-encoder is available.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)

Document = Dict[str, Any]


class RerankAgent(BaseAgent):
    """Reranks retrieved documents by relevance to the query.

    Parameters
    ----------
    config:
        System configuration.  ``config.retrieval.rerank_top_k`` controls
        how many top documents are kept after reranking.
    cross_encoder_model:
        Optional name of a sentence-transformers cross-encoder model,
        e.g. ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
    """

    _BATCH_SIZE = 5  # Documents scored per LLM call

    _SYSTEM_PROMPT = (
        "You are a relevance-scoring expert for a retrieval-augmented generation system.\n"
        "Given a user query and a list of documents, score each document's relevance\n"
        "to the query on a scale from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).\n\n"
        "Return a JSON array of objects: [{\"index\": 0, \"score\": 0.85}, ...]\n"
        "Index corresponds to the 0-based position in the document list.\n"
        "Be strict: only score ≥ 0.6 for documents that genuinely help answer the query.\n"
    )

    def __init__(
        self,
        config: SystemConfig,
        cross_encoder_model: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._cross_encoder: Optional[Any] = None
        if cross_encoder_model:
            self._cross_encoder = self._load_cross_encoder(cross_encoder_model)

    @property
    def name(self) -> str:
        return "rerank"

    @property
    def description(self) -> str:
        return (
            "Scores retrieved documents for relevance, filters below-threshold ones, "
            "and returns a ranked list."
        )

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        docs = state.get("retrieved_documents") or []
        if not docs:
            logger.info("[rerank] No documents to rerank.")
            return self._update_state(state, {"reranked_documents": []})

        query = state.get("rewritten_query") or state.get("query", "")
        threshold = self.config.retrieval.similarity_threshold
        top_k = self.config.retrieval.rerank_top_k

        # Try cross-encoder first (cheap), fall back to LLM
        if self._cross_encoder is not None:
            scored = self._cross_encoder_score(query, docs)
        else:
            scored = self._llm_score(query, docs)

        # Filter and sort
        filtered = [d for d in scored if d["rerank_score"] >= threshold]
        filtered.sort(key=lambda d: d["rerank_score"], reverse=True)
        reranked = filtered[:top_k]

        logger.info(
            "[rerank] %d → %d docs after reranking (threshold=%.2f)",
            len(docs),
            len(reranked),
            threshold,
        )
        self._log_exit(state)
        return self._update_state(state, {"reranked_documents": reranked})

    # ── Cross-encoder scoring ─────────────────────────────────────────────

    @staticmethod
    def _load_cross_encoder(model_name: str) -> Optional[Any]:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("[rerank] Loading cross-encoder: %s", model_name)
            return CrossEncoder(model_name)
        except Exception as exc:
            logger.warning("[rerank] Failed to load cross-encoder: %s", exc)
            return None

    def _cross_encoder_score(self, query: str, docs: List[Document]) -> List[Document]:
        try:
            pairs = [(query, d["content"]) for d in docs]
            scores = self._cross_encoder.predict(pairs)
            result = []
            for doc, score in zip(docs, scores):
                result.append({**doc, "rerank_score": float(score)})
            return result
        except Exception as exc:
            logger.warning("[rerank] Cross-encoder scoring failed: %s; falling back to LLM.", exc)
            return self._llm_score(query, docs)

    # ── LLM scoring ───────────────────────────────────────────────────────

    def _llm_score(self, query: str, docs: List[Document]) -> List[Document]:
        """Score documents in batches using the LLM."""
        scored: List[Document] = []
        for batch_start in range(0, len(docs), self._BATCH_SIZE):
            batch = docs[batch_start : batch_start + self._BATCH_SIZE]
            batch_scored = self._score_batch(query, batch, offset=batch_start)
            scored.extend(batch_scored)
        return scored

    def _score_batch(
        self, query: str, batch: List[Document], offset: int = 0
    ) -> List[Document]:
        doc_list = "\n\n".join(
            f"[{i}] {doc['content'][:500]}" for i, doc in enumerate(batch)
        )
        user_content = f"Query: {query}\n\nDocuments:\n{doc_list}"

        messages = [
            SystemMessage(content=self._SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = self._call_llm(messages)
            score_data = self._parse_scores(response.content)
        except Exception as exc:
            logger.warning("[rerank] LLM scoring failed: %s; assigning neutral scores.", exc)
            score_data = {i: 0.5 for i in range(len(batch))}

        result = []
        for local_idx, doc in enumerate(batch):
            rerank_score = score_data.get(local_idx, doc.get("score", 0.5))
            result.append({**doc, "rerank_score": float(rerank_score)})
        return result

    @staticmethod
    def _parse_scores(content: str) -> Dict[int, float]:
        """Parse LLM JSON score output into {local_index: score}."""
        try:
            # Try to extract JSON array from response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                return {}
            data = json.loads(content[start:end])
            return {item["index"]: float(item["score"]) for item in data}
        except Exception:
            return {}
