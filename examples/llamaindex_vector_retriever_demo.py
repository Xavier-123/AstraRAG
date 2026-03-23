"""
examples/llamaindex_vector_retriever_demo.py
---------------------------------------------
Demonstrates the LlamaIndex + FAISS vector retriever.

Run::

    python examples/llamaindex_vector_retriever_demo.py

No API key is required — the demo uses a local HuggingFace embedding model.
"""

from __future__ import annotations

import json
import logging
import os
import sys

# Ensure the project root is on the Python path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("llamaindex_vector_retriever_demo")

PERSIST_DIR = "/tmp/demo_llamaindex_faiss"
RESULTS_JSON = "/tmp/demo_llamaindex_faiss/results.json"


def main() -> None:
    from astra_rag.agents.llamaindex_vector_retriever import LlamaIndexVectorRetriever

    # ── 1. Create the retriever ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Initialise retriever (FAISS + local embeddings)")
    print("=" * 60)

    retriever = LlamaIndexVectorRetriever(
        persist_dir=PERSIST_DIR,
        chunk_size=256,
        chunk_overlap=30,
        embed_dim=384,  # BAAI/bge-small-en-v1.5 dimension
    )

    # ── 2. Prepare sample texts ────────────────────────────────────────────
    texts = [
        (
            "Retrieval-Augmented Generation (RAG) is an AI framework that enhances "
            "large language model outputs by incorporating external knowledge retrieval. "
            "It combines a retriever component that fetches relevant documents from a "
            "knowledge base with a generator that produces answers grounded in those "
            "documents, significantly reducing hallucinations."
        ),
        (
            "FAISS (Facebook AI Similarity Search) is a library developed by Meta AI "
            "for efficient similarity search and clustering of dense vectors. It supports "
            "several index types including flat (exact) search, IVF (inverted file), and "
            "HNSW (hierarchical navigable small world) graphs for approximate nearest "
            "neighbor search at scale."
        ),
        (
            "LlamaIndex is a data framework for LLM applications that provides tools "
            "for ingesting, structuring, and accessing private or domain-specific data. "
            "It offers connectors to various data sources, index structures for efficient "
            "retrieval, and query engines that combine retrieval with LLM synthesis."
        ),
        (
            "Vector embeddings are numerical representations of data in high-dimensional "
            "space. Text embeddings capture semantic meaning so that similar texts have "
            "nearby vectors. Popular embedding models include OpenAI text-embedding-3, "
            "BGE, and Sentence-Transformers. Dimensionality typically ranges from 384 to "
            "3072 depending on the model."
        ),
        (
            "The transformer architecture, introduced in the 'Attention Is All You Need' "
            "paper, is the foundation of modern large language models. It uses multi-head "
            "self-attention to process input tokens in parallel, enabling efficient training "
            "on large datasets. Key components include the encoder, decoder, positional "
            "encoding, and layer normalisation."
        ),
    ]

    metadatas = [
        {"source": "rag_overview.md", "topic": "RAG"},
        {"source": "faiss_docs.md", "topic": "FAISS"},
        {"source": "llamaindex_docs.md", "topic": "LlamaIndex"},
        {"source": "embeddings_guide.md", "topic": "Embeddings"},
        {"source": "transformers_paper.md", "topic": "Transformers"},
    ]

    # ── 3. Build the index ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Build FAISS index from sample texts")
    print("=" * 60)

    num_chunks = retriever.build_index(texts, metadatas=metadatas)
    print(f"  Indexed {num_chunks} chunks from {len(texts)} documents.")

    # ── 4. Single-query retrieval ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Single-query retrieval")
    print("=" * 60)

    query = "How does RAG reduce hallucinations?"
    results = retriever.retrieve(query, top_k=3)

    print(f"\n  Query: {query}")
    print(f"  Top-{len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n  [{i}] Score: {doc['score']:.4f}")
        print(f"      Chunk ID: {doc['metadata'].get('chunk_id', 'N/A')}")
        print(f"      Source:   {doc['metadata'].get('source', 'N/A')}")
        print(f"      Content:  {doc['content'][:120]}...")

    # ── 5. Batch retrieval ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Batch retrieval (multiple queries)")
    print("=" * 60)

    batch_queries = [
        "What is FAISS?",
        "Explain vector embeddings",
        "What is the transformer architecture?",
    ]
    batch_results = retriever.batch_retrieve(batch_queries, top_k=2)

    for q, docs in batch_results.items():
        print(f"\n  Query: {q}")
        for i, doc in enumerate(docs, 1):
            print(f"    [{i}] Score={doc['score']:.4f} | {doc['content'][:80]}...")

    # ── 6. Save results to JSON ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Save results to JSON")
    print("=" * 60)

    retriever.save_results_to_json(results, RESULTS_JSON)
    print(f"  Results saved to {RESULTS_JSON}")

    # Show the JSON content
    with open(RESULTS_JSON, encoding="utf-8") as fp:
        saved = json.load(fp)
    print(f"  JSON contains {len(saved)} result entries.")
    print(f"  Sample entry keys: {list(saved[0].keys())}")

    # ── 7. Reload from persisted index ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: Reload persisted index and re-query")
    print("=" * 60)

    retriever2 = LlamaIndexVectorRetriever(
        persist_dir=PERSIST_DIR,
        chunk_size=256,
        chunk_overlap=30,
        embed_dim=384,
    )
    reloaded_results = retriever2.retrieve("What is LlamaIndex?", top_k=2)
    print(f"\n  Re-query after reload: {len(reloaded_results)} results")
    for i, doc in enumerate(reloaded_results, 1):
        print(f"    [{i}] Score={doc['score']:.4f} | {doc['content'][:80]}...")

    print("\n" + "=" * 60)
    print("Demo completed successfully ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
