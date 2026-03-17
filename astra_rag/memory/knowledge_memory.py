"""
astra_rag/memory/knowledge_memory.py
--------------------------------------
KnowledgeMemory — abstract vector knowledge-base interface.

Supported backends
------------------
``chroma`` (default)
    Persistent ChromaDB collection stored on disk.

``faiss``
    In-process FAISS index (fast, no server required).
    Serialised to disk on ``save()`` / loaded on ``load()``.

``milvus`` / ``weaviate``
    Stubs — inject a pre-built LangChain vectorstore to use these backends.

Public API
----------
``add_documents(documents)``
    Embed and store a list of ``{"content": str, "metadata": dict}`` dicts.

``search(query, top_k) → List[Dict]``
    Return top-k nearest documents with scores.

``delete(doc_ids)``
    Remove documents by ID.

``save()`` / ``load()``
    Persist / restore the index (FAISS only; Chroma auto-persists).
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from astra_rag.core.config import SystemConfig

logger = logging.getLogger(__name__)

Document = Dict[str, Any]


# ── Abstract interface ────────────────────────────────────────────────────────


class BaseKnowledgeBackend(ABC):
    """Abstract backend that all vector stores must implement."""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents. Returns assigned document IDs."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Document]:
        """Semantic search. Returns list of {content, score, metadata} dicts."""

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> None:
        """Remove documents by ID."""


# ── Chroma backend ────────────────────────────────────────────────────────────


class _ChromaBackend(BaseKnowledgeBackend):
    def __init__(self, config: SystemConfig, embeddings: Any) -> None:
        try:
            import chromadb
            from langchain_community.vectorstores import Chroma

            self._store = Chroma(
                collection_name="astra_rag_knowledge",
                embedding_function=embeddings,
                persist_directory=config.memory.chroma_persist_dir,
            )
            logger.info("[KnowledgeMemory] Using Chroma backend.")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Chroma: {exc}") from exc

    def add_documents(self, documents: List[Document]) -> List[str]:
        from langchain_core.documents import Document as LCDoc

        lc_docs = [
            LCDoc(page_content=d["content"], metadata=d.get("metadata", {}))
            for d in documents
        ]
        ids = [str(uuid.uuid4()) for _ in lc_docs]
        self._store.add_documents(lc_docs, ids=ids)
        return ids

    def search(self, query: str, top_k: int) -> List[Document]:
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            {"content": doc.page_content, "score": float(score), "metadata": doc.metadata}
            for doc, score in results
        ]

    def delete(self, doc_ids: List[str]) -> None:
        self._store.delete(ids=doc_ids)


# ── FAISS backend ─────────────────────────────────────────────────────────────


class _FAISSBackend(BaseKnowledgeBackend):
    def __init__(self, config: SystemConfig, embeddings: Any) -> None:
        try:
            from langchain_community.vectorstores import FAISS

            self._embeddings = embeddings
            self._index_path = config.memory.faiss_index_path
            try:
                self._store = FAISS.load_local(
                    self._index_path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("[KnowledgeMemory] Loaded existing FAISS index.")
            except Exception:
                logger.info("[KnowledgeMemory] Creating new FAISS index.")
                # Bootstrap with a dummy document — FAISS requires at least one
                from langchain_core.documents import Document as LCDoc

                self._store = FAISS.from_documents(
                    [LCDoc(page_content="init", metadata={})], embeddings
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise FAISS: {exc}") from exc

    def add_documents(self, documents: List[Document]) -> List[str]:
        from langchain_core.documents import Document as LCDoc

        lc_docs = [
            LCDoc(page_content=d["content"], metadata=d.get("metadata", {}))
            for d in documents
        ]
        ids = self._store.add_documents(lc_docs)
        return ids

    def search(self, query: str, top_k: int) -> List[Document]:
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            {"content": doc.page_content, "score": float(score), "metadata": doc.metadata}
            for doc, score in results
        ]

    def delete(self, doc_ids: List[str]) -> None:
        self._store.delete(doc_ids)

    def save(self) -> None:
        self._store.save_local(self._index_path)


# ── Injection backend (Milvus / Weaviate / custom) ────────────────────────────


class _InjectedBackend(BaseKnowledgeBackend):
    """Wraps a pre-built LangChain vectorstore object."""

    def __init__(self, vectorstore: Any) -> None:
        self._store = vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        from langchain_core.documents import Document as LCDoc

        lc_docs = [
            LCDoc(page_content=d["content"], metadata=d.get("metadata", {}))
            for d in documents
        ]
        return self._store.add_documents(lc_docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        return [
            {"content": doc.page_content, "score": float(score), "metadata": doc.metadata}
            for doc, score in results
        ]

    def delete(self, doc_ids: List[str]) -> None:
        self._store.delete(doc_ids)


# ── Public class ──────────────────────────────────────────────────────────────


class KnowledgeMemory:
    """Vector knowledge base with swappable backends.

    Parameters
    ----------
    config:
        System configuration.
    vectorstore:
        Optional pre-built LangChain vectorstore. When provided, bypasses
        the Chroma/FAISS auto-initialisation.
    """

    def __init__(
        self,
        config: SystemConfig,
        vectorstore: Optional[Any] = None,
    ) -> None:
        from astra_rag.utils.llm import get_embeddings

        self._embeddings = get_embeddings(config)

        if vectorstore is not None:
            self._backend: BaseKnowledgeBackend = _InjectedBackend(vectorstore)
        elif config.memory.vector_db_type == "faiss":
            self._backend = _FAISSBackend(config, self._embeddings)
        else:
            try:
                self._backend = _ChromaBackend(config, self._embeddings)
            except Exception as exc:
                logger.warning(
                    "[KnowledgeMemory] Chroma init failed (%s); falling back to FAISS.", exc
                )
                self._backend = _FAISSBackend(config, self._embeddings)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and persist *documents*.

        Parameters
        ----------
        documents:
            List of ``{"content": str, "metadata": dict}`` dicts.

        Returns
        -------
        List[str]
            Assigned document IDs.
        """
        return self._backend.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Semantic search returning top-k documents with relevance scores."""
        return self._backend.search(query, top_k)

    def delete(self, doc_ids: List[str]) -> None:
        """Remove documents by their IDs."""
        self._backend.delete(doc_ids)

    def save(self) -> None:
        """Persist the index to disk (FAISS backend only)."""
        if hasattr(self._backend, "save"):
            self._backend.save()
