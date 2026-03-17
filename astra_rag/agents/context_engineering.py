"""
astra_rag/agents/context_engineering.py
-----------------------------------------
ContextEngineeringAgent — Pipeline position: node 6 of 8.

Responsibilities
----------------
Transform the raw ``reranked_documents`` list into a high-quality,
token-budget-aware context string ready for injection into the reasoning
prompt.

Processing pipeline (in order)
--------------------------------
1. **Deduplication** — removes near-duplicate passages using MD5 fingerprints
   of the first 200 characters (O(n) pass).
2. **Relevance filtering** — discards documents with ``rerank_score < threshold``
   (second safety net after reranking).
3. **Compression** — long documents (> ``max_doc_tokens`` tokens) are
   summarised by the LLM to their key facts.
4. **Context ordering** — most-relevant documents appear first so the LLM's
   recency bias works in our favour.
5. **Token-budget assembly** — appends documents to the context string until
   ``config.retrieval.max_context_tokens`` would be exceeded, then stops.
6. **Summarisation fallback** — if even the top document exceeds budget, the
   whole context is summarised.

Output
------
``state["context"]``: formatted multi-document context string ready for
the reasoning prompt.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState
from astra_rag.utils.llm import count_tokens, truncate_to_token_limit

logger = logging.getLogger(__name__)

Document = Dict[str, Any]

_MAX_DOC_TOKENS = 600  # Documents longer than this are compressed
_MIN_RERANK_SCORE = 0.3  # Hard floor for inclusion


class ContextEngineeringAgent(BaseAgent):
    """Deduplicates, compresses, and assembles the final context string."""

    _COMPRESSION_SYSTEM = (
        "You are a precise document compressor. Given a long document, extract and "
        "preserve ONLY the facts directly relevant to the query. "
        "Be concise — target ≤ 150 words. Do not add any new information."
    )

    _SUMMARISE_SYSTEM = (
        "You are a context summariser. Summarise the following multi-document context "
        "into a coherent, dense paragraph that preserves ALL key facts needed to answer "
        "the query. Target ≤ 300 words."
    )

    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "context_engineering"

    @property
    def description(self) -> str:
        return (
            "Deduplicates, compresses, and assembles reranked documents into a "
            "token-budget-aware context string for the reasoning prompt."
        )

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        docs = state.get("reranked_documents") or state.get("retrieved_documents") or []
        query = state.get("rewritten_query") or state.get("query", "")
        budget = self.config.retrieval.max_context_tokens

        if not docs:
            logger.info("[context_engineering] No documents; empty context.")
            return self._update_state(state, {"context": ""})

        # 1. Relevance filter
        docs = [d for d in docs if d.get("rerank_score", d.get("score", 1.0)) >= _MIN_RERANK_SCORE]

        # 2. Deduplication
        docs = self._deduplicate(docs)

        # 3. Sort by relevance (highest first)
        docs.sort(key=lambda d: d.get("rerank_score", d.get("score", 0.0)), reverse=True)

        # 4. Compress long docs
        docs = [self._maybe_compress(doc, query) for doc in docs]

        # 5. Assemble within token budget
        context = self._assemble(docs, budget, query)

        logger.info(
            "[context_engineering] Context assembled: %d tokens from %d docs",
            count_tokens(context, self.config.llm.model),
            len(docs),
        )
        self._log_exit(state)
        return self._update_state(state, {"context": context})

    # ── Processing steps ──────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(docs: List[Document]) -> List[Document]:
        seen: set = set()
        unique: List[Document] = []
        for doc in docs:
            fingerprint = hashlib.md5(doc["content"][:200].encode()).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(doc)
        return unique

    def _maybe_compress(self, doc: Document, query: str) -> Document:
        """Return a compressed copy of *doc* if it exceeds the token ceiling."""
        content = doc.get("content", "")
        token_count = count_tokens(content, self.config.llm.model)
        if token_count <= _MAX_DOC_TOKENS:
            return doc

        try:
            messages = [
                SystemMessage(content=self._COMPRESSION_SYSTEM),
                HumanMessage(content=f"Query: {query}\n\nDocument:\n{content}"),
            ]
            compressed = self._call_llm(messages).content
            return {**doc, "content": compressed, "compressed": True}
        except Exception as exc:
            logger.warning("[context_engineering] Compression failed: %s; truncating.", exc)
            truncated = truncate_to_token_limit(content, _MAX_DOC_TOKENS, self.config.llm.model)
            return {**doc, "content": truncated, "truncated": True}

    def _assemble(self, docs: List[Document], budget: int, query: str) -> str:
        """Concatenate doc content within *budget* tokens; summarise if needed."""
        parts: List[str] = []
        used_tokens = 0

        for i, doc in enumerate(docs):
            source = doc.get("source", "unknown")
            score = doc.get("rerank_score", doc.get("score", 0.0))
            header = f"[Source {i + 1}: {source} | relevance={score:.2f}]"
            content = doc.get("content", "")
            block = f"{header}\n{content}"
            block_tokens = count_tokens(block, self.config.llm.model)

            if used_tokens + block_tokens > budget:
                if not parts:
                    # Even the first doc is too long — summarise it
                    block = self._summarise_single(content, query, budget)
                    parts.append(block)
                break

            parts.append(block)
            used_tokens += block_tokens

        context = "\n\n---\n\n".join(parts)

        # Final safety net: if assembled context is still too long, summarise
        if count_tokens(context, self.config.llm.model) > budget:
            context = self._summarise_context(context, query, budget)

        return context

    def _summarise_single(self, content: str, query: str, budget: int) -> str:
        try:
            messages = [
                SystemMessage(content=self._COMPRESSION_SYSTEM),
                HumanMessage(content=f"Query: {query}\n\nDocument:\n{content}"),
            ]
            return self._call_llm(messages).content
        except Exception:
            return truncate_to_token_limit(content, budget, self.config.llm.model)

    def _summarise_context(self, context: str, query: str, budget: int) -> str:
        try:
            messages = [
                SystemMessage(content=self._SUMMARISE_SYSTEM),
                HumanMessage(content=f"Query: {query}\n\nContext:\n{context}"),
            ]
            return self._call_llm(messages).content
        except Exception:
            return truncate_to_token_limit(context, budget, self.config.llm.model)
