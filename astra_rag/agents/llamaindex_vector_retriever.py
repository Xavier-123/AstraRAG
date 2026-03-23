"""
astra_rag/agents/llamaindex_vector_retriever.py
-------------------------------------------------
LlamaIndex-based FAISS vector retriever — implements ``BaseRetriever``.

This module provides a complete vector retrieval pipeline built on LlamaIndex:

1. **Chunking**   – Splits raw text into overlapping chunks via
   ``SentenceSplitter``.
2. **Indexing**   – Builds a FAISS flat-L2 index through LlamaIndex's
   ``FaissVectorStore``.
3. **Retrieval**  – Performs top-k similarity search and returns normalised
   ``Document`` dicts that are compatible with ``MultiRetrieverAgent``.
4. **Persistence** – Saves the FAISS index to disk and writes retrieved
   results (chunk text, score, metadata) to a JSON file.

Usage
-----
See ``examples/llamaindex_vector_retriever_demo.py`` for a runnable demo.

Extending embeddings
--------------------
Pass any LlamaIndex ``BaseEmbedding`` instance to the constructor.  By
default the class uses HuggingFace ``BAAI/bge-small-en-v1.5`` which runs
locally without an API key.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import faiss
from llama_index.core import Document as LIDocument
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore

from astra_rag.agents.multi_retriever import BaseRetriever, Document

logger = logging.getLogger(__name__)


def _default_embed_model() -> Any:
    """Return a lightweight local embedding model (no API key required)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


class LlamaIndexVectorRetriever(BaseRetriever):
    """LlamaIndex + FAISS vector retriever.

    Parameters
    ----------
    persist_dir:
        Directory where the FAISS index and JSON results are saved.
    embed_model:
        A LlamaIndex ``BaseEmbedding`` instance.  When ``None``, falls back
        to a local HuggingFace model (``BAAI/bge-small-en-v1.5``).
    chunk_size:
        Target token count per chunk.
    chunk_overlap:
        Overlap tokens between consecutive chunks.
    embed_dim:
        Dimensionality of the embedding vectors.  Must match the model.
        Defaults to 384 (BGE-small).
    """

    def __init__(
        self,
        persist_dir: str = "./faiss_llamaindex_store",
        embed_model: Optional[Any] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embed_dim: int = 384,
    ) -> None:
        self._persist_dir = persist_dir
        self._embed_model = embed_model or _default_embed_model()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embed_dim = embed_dim

        # Apply embedding model globally for LlamaIndex
        Settings.embed_model = self._embed_model

        # Node parser for chunking
        self._parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Attempt to load an existing index, otherwise prepare for building
        self._index: Optional[VectorStoreIndex] = None
        self._load_index()

    # ── BaseRetriever interface ────────────────────────────────────────────

    @property
    def source_name(self) -> str:  # noqa: D401
        return "llamaindex_vector"

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Return the *top_k* most similar chunks for *query*.

        Each result is a normalised ``Document`` dict with keys
        ``id``, ``content``, ``source``, ``score``, and ``metadata``.
        """
        if self._index is None:
            logger.warning(
                "[LlamaIndexVectorRetriever] No index built yet — call build_index() first."
            )
            return []

        try:
            retriever = self._index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            docs: List[Document] = []
            for node_with_score in nodes:
                node = node_with_score.node
                docs.append(
                    self._make_doc(
                        content=node.get_content(),
                        source="llamaindex_vector",
                        score=float(node_with_score.score) if node_with_score.score else 0.0,
                        metadata={
                            "chunk_id": node.node_id,
                            **node.metadata,
                        },
                    )
                )
            return docs
        except Exception as exc:
            logger.error("[LlamaIndexVectorRetriever] Retrieval failed: %s", exc)
            return []

    # ── Index building ─────────────────────────────────────────────────────

    def build_index(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Chunk *texts*, embed them, and build a FAISS index.

        Parameters
        ----------
        texts:
            Raw text strings to be chunked and indexed.
        metadatas:
            Optional per-text metadata dicts.  Each dict is attached to every
            chunk produced from the corresponding text.

        Returns
        -------
        int
            Total number of chunks (nodes) indexed.
        """
        documents: List[LIDocument] = []
        for i, text in enumerate(texts):
            meta = (metadatas[i] if metadatas and i < len(metadatas) else {}).copy()
            meta.setdefault("source_index", i)
            documents.append(LIDocument(text=text, metadata=meta))

        # Chunk documents into nodes
        nodes = self._parser.get_nodes_from_documents(documents)
        for node in nodes:
            if not node.node_id:
                node.node_id = str(uuid.uuid4())

        logger.info(
            "[LlamaIndexVectorRetriever] Chunked %d texts → %d nodes (chunk_size=%d).",
            len(texts),
            len(nodes),
            self._chunk_size,
        )

        # Build FAISS vector store and index
        faiss_index = faiss.IndexFlatL2(self._embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )

        # Persist to disk
        self._persist_index()
        return len(nodes)

    # ── Batch retrieval ────────────────────────────────────────────────────

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> Dict[str, List[Document]]:
        """Run retrieval for multiple queries.

        Parameters
        ----------
        queries:
            List of query strings.
        top_k:
            Maximum number of results per query.

        Returns
        -------
        Dict[str, List[Document]]
            Mapping from query string to its result list.
        """
        results: Dict[str, List[Document]] = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k=top_k)
        return results

    # ── Persistence helpers ────────────────────────────────────────────────

    def _persist_index(self) -> None:
        """Save the FAISS index and LlamaIndex storage to *persist_dir*."""
        if self._index is None:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        self._index.storage_context.persist(persist_dir=self._persist_dir)
        logger.info("[LlamaIndexVectorRetriever] Index persisted to %s", self._persist_dir)

    def _load_index(self) -> None:
        """Attempt to load a previously persisted index from *persist_dir*."""
        vector_store_path = os.path.join(self._persist_dir, "default__vector_store.json")
        if not os.path.exists(vector_store_path):
            logger.debug(
                "[LlamaIndexVectorRetriever] No existing index at %s", self._persist_dir
            )
            return
        try:
            vector_store = FaissVectorStore.from_persist_dir(self._persist_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=self._persist_dir,
            )
            self._index = load_index_from_storage(storage_context=storage_context)
            logger.info(
                "[LlamaIndexVectorRetriever] Loaded existing index from %s", self._persist_dir
            )
        except Exception as exc:
            logger.warning(
                "[LlamaIndexVectorRetriever] Failed to load index (%s); "
                "call build_index() to create a new one.",
                exc,
            )

    def save_results_to_json(
        self,
        results: List[Document],
        output_path: str,
    ) -> None:
        """Persist retrieval results to a JSON file.

        Parameters
        ----------
        results:
            List of ``Document`` dicts returned by ``retrieve`` or
            ``batch_retrieve``.
        output_path:
            File path for the output JSON.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        logger.info("[LlamaIndexVectorRetriever] Results saved to %s", output_path)
