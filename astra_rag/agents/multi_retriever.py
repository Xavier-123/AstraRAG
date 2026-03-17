"""
astra_rag/agents/multi_retriever.py
-------------------------------------
MultiRetrieverAgent — Pipeline position: node 4 of 8.

Architecture
------------
The agent owns a registry of *retriever backends* — each implementing the
abstract ``BaseRetriever`` interface.  At runtime it reads
``state["retrieval_strategy"]`` and activates the matching backends,
running them concurrently with ``asyncio.gather`` and aggregating results.

Included retriever implementations
-----------------------------------
``VectorRetriever``
    Queries a vector knowledge store (Chroma / FAISS via KnowledgeMemory).
    Falls back to returning an empty list when no store is configured.

``GraphRetriever``
    Abstract stub for a knowledge-graph backend (Neo4j, etc.).
    Real implementations inject a graph client via the constructor.

``WebRetriever``
    Abstract stub for live internet search (Tavily, SerpAPI, DuckDuckGo).
    Real implementations inject a search client.

Adding a new retriever
-----------------------
1. Subclass ``BaseRetriever``.
2. Implement ``retrieve(query, top_k) → List[Document]``.
3. Register an instance in ``MultiRetrieverAgent._build_registry``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)

# ── Document type alias ────────────────────────────────────────────────────────
Document = Dict[str, Any]
"""
Normalised document dict:
    {
        "id":      str,       # deterministic hash of source+content
        "content": str,
        "source":  str,       # e.g. "vector_db", "web", "graph"
        "score":   float,     # similarity / relevance score 0-1
        "metadata": dict      # original metadata from the backend
    }
"""


# ── Abstract base retriever ────────────────────────────────────────────────────


class BaseRetriever(ABC):
    """Abstract interface for a single-source retriever backend."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier, e.g. 'vector', 'graph', 'web'."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Document]:
        """Retrieve documents synchronously.

        Parameters
        ----------
        query:
            The (rewritten) query string.
        top_k:
            Maximum number of documents to return.
        """

    async def aretrieve(self, query: str, top_k: int) -> List[Document]:
        """Async wrapper — override for genuine async I/O."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, top_k)

    # ── Normalisation helper ───────────────────────────────────────────────

    @staticmethod
    def _make_doc(
        content: str,
        source: str,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        doc_id = hashlib.md5(f"{source}::{content[:200]}".encode()).hexdigest()
        return {
            "id": doc_id,
            "content": content,
            "source": source,
            "score": score,
            "metadata": metadata or {},
        }


# ── Concrete retriever: Vector ────────────────────────────────────────────────


class VectorRetriever(BaseRetriever):
    """Retrieves documents from a vector knowledge store.

    Parameters
    ----------
    knowledge_memory:
        An instance of ``KnowledgeMemory``; if ``None``, returns empty list.
    """

    def __init__(self, knowledge_memory: Optional[Any] = None) -> None:
        self._km = knowledge_memory

    @property
    def source_name(self) -> str:
        return "vector"

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self._km is None:
            logger.debug("[VectorRetriever] No knowledge memory configured, returning []")
            return []
        try:
            results = self._km.search(query, top_k=top_k)
            return [
                self._make_doc(
                    content=r.get("content", ""),
                    source="vector",
                    score=r.get("score", 0.0),
                    metadata=r.get("metadata", {}),
                )
                for r in results
            ]
        except Exception as exc:
            logger.error("[VectorRetriever] Retrieval failed: %s", exc)
            return []


# ── Concrete retriever: Graph ─────────────────────────────────────────────────


class GraphRetriever(BaseRetriever):
    """Knowledge-graph retriever stub.

    Inject a real graph client (e.g. ``Neo4jGraph`` from
    ``langchain_community``) via the constructor to activate this backend.
    """

    def __init__(self, graph_client: Optional[Any] = None) -> None:
        self._client = graph_client

    @property
    def source_name(self) -> str:
        return "graph"

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self._client is None:
            logger.debug("[GraphRetriever] No graph client configured, returning []")
            return []
        try:
            raw = self._client.query(query, top_k=top_k)
            return [
                self._make_doc(
                    content=str(item),
                    source="graph",
                    score=0.8,
                    metadata={"graph_item": item},
                )
                for item in raw[:top_k]
            ]
        except Exception as exc:
            logger.error("[GraphRetriever] Query failed: %s", exc)
            return []


# ── Concrete retriever: Web ───────────────────────────────────────────────────


class WebRetriever(BaseRetriever):
    """Live internet search retriever stub.

    Inject a real search client (e.g. ``TavilySearchAPIWrapper`` or
    ``DuckDuckGoSearchAPIWrapper``) to activate this backend.
    """

    def __init__(self, search_client: Optional[Any] = None) -> None:
        self._client = search_client

    @property
    def source_name(self) -> str:
        return "web"

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self._client is None:
            logger.debug("[WebRetriever] No search client configured, returning []")
            return []
        try:
            results = self._client.results(query, max_results=top_k)
            docs: List[Document] = []
            for r in results[:top_k]:
                snippet = r.get("snippet") or r.get("body") or r.get("content", "")
                docs.append(
                    self._make_doc(
                        content=snippet,
                        source="web",
                        score=0.7,
                        metadata={"url": r.get("link") or r.get("url", "")},
                    )
                )
            return docs
        except Exception as exc:
            logger.error("[WebRetriever] Search failed: %s", exc)
            return []


# ── Main agent ────────────────────────────────────────────────────────────────


class MultiRetrieverAgent(BaseAgent):
    """Dispatches to one or more retriever backends and aggregates results.

    Parameters
    ----------
    config:
        System configuration.
    knowledge_memory:
        Optional pre-built ``KnowledgeMemory`` instance.
    graph_client:
        Optional graph DB client.
    search_client:
        Optional web search client.
    """

    def __init__(
        self,
        config: SystemConfig,
        knowledge_memory: Optional[Any] = None,
        graph_client: Optional[Any] = None,
        search_client: Optional[Any] = None,
    ) -> None:
        super().__init__(config)
        self._registry: Dict[str, BaseRetriever] = {
            "vector": VectorRetriever(knowledge_memory),
            "graph": GraphRetriever(graph_client),
            "web": WebRetriever(search_client),
        }

    @property
    def name(self) -> str:
        return "multi_retriever"

    @property
    def description(self) -> str:
        return (
            "Executes vector, graph, and/or web retrievers in parallel "
            "based on the planned strategy, then deduplicates results."
        )

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        strategy = state.get("retrieval_strategy", "vector")
        if strategy == "none":
            return self._update_state(state, {"retrieved_documents": []})

        query = state.get("rewritten_query") or state.get("query", "")
        top_k = self.config.retrieval.top_k

        active_sources = self._resolve_sources(strategy)
        logger.info("[multi_retriever] Active sources: %s", active_sources)

        # Run retrievers (use asyncio if available, otherwise sequential)
        all_docs: List[Document] = []
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in an async context — schedule coroutines
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    futures = [
                        pool.submit(self._registry[src].retrieve, query, top_k)
                        for src in active_sources
                        if src in self._registry
                    ]
                    for f in futures:
                        all_docs.extend(f.result())
            else:
                all_docs = loop.run_until_complete(
                    self._gather(query, top_k, active_sources)
                )
        except RuntimeError:
            # No event loop
            for src in active_sources:
                if src in self._registry:
                    all_docs.extend(self._registry[src].retrieve(query, top_k))

        deduped = self._deduplicate(all_docs)
        logger.info("[multi_retriever] Retrieved %d docs (after dedup)", len(deduped))
        self._log_exit(state)
        return self._update_state(state, {"retrieved_documents": deduped})

    async def arun(self, state: GraphState) -> GraphState:
        self._log_entry(state)
        strategy = state.get("retrieval_strategy", "vector")
        if strategy == "none":
            return self._update_state(state, {"retrieved_documents": []})

        query = state.get("rewritten_query") or state.get("query", "")
        top_k = self.config.retrieval.top_k
        active_sources = self._resolve_sources(strategy)

        all_docs = await self._gather(query, top_k, active_sources)
        deduped = self._deduplicate(all_docs)
        self._log_exit(state)
        return self._update_state(state, {"retrieved_documents": deduped})

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _gather(
        self, query: str, top_k: int, sources: List[str]
    ) -> List[Document]:
        tasks = [
            self._registry[src].aretrieve(query, top_k)
            for src in sources
            if src in self._registry
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        docs: List[Document] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("[multi_retriever] Retriever error: %s", r)
            else:
                docs.extend(r)
        return docs

    @staticmethod
    def _resolve_sources(strategy: str) -> List[str]:
        mapping: Dict[str, List[str]] = {
            "vector": ["vector"],
            "graph": ["graph"],
            "web": ["web"],
            "hybrid": ["vector", "graph"],
            "hybrid_web": ["vector", "graph", "web"],
        }
        return mapping.get(strategy, ["vector"])

    @staticmethod
    def _deduplicate(docs: List[Document]) -> List[Document]:
        """Remove documents with identical IDs, keeping the highest score."""
        seen: Dict[str, Document] = {}
        for doc in docs:
            doc_id = doc["id"]
            if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
                seen[doc_id] = doc
        return sorted(seen.values(), key=lambda d: d["score"], reverse=True)
