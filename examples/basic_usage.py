"""
examples/basic_usage.py
------------------------
Complete walkthrough of the AstraRAG framework.

Prerequisites
-------------
1. Install dependencies:
       pip install -r requirements.txt

2. Set your OpenAI API key:
       export OPENAI_API_KEY="sk-..."

   Or create a .env file in the project root:
       OPENAI_API_KEY=sk-...

3. (Optional) Start Redis for persistent conversation memory:
       docker run -d -p 6379:6379 redis:alpine

Run this file:
       python examples/basic_usage.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Ensure the project root is on the Python path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("basic_usage")


# =============================================================================
# 1. System initialisation
# =============================================================================

def demo_basic_query() -> None:
    """Run a simple factual query through the full pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Query")
    print("=" * 60)

    from astra_rag import AstraRAGSystem
    from astra_rag.core.config import LLMConfig, RetrievalConfig, SystemConfig

    # Customise configuration
    config = SystemConfig(
        llm=LLMConfig(
            model="gpt-4o-mini",   # Use a cheaper model for demos
            temperature=0.0,        # Deterministic outputs
            max_tokens=1024,
        ),
        retrieval=RetrievalConfig(
            top_k=3,               # Retrieve 3 documents per strategy
            similarity_threshold=0.4,
            rerank_top_k=2,
        ),
        max_reflection_iterations=1,  # Quick demo — limit reflection loops
        debug=False,
    )

    # Initialise the system
    # For this demo no vector store is wired up, so retrieval returns empty
    # results and the LLM answers from its parametric knowledge.
    system = AstraRAGSystem(config=config, enable_self_improving=True)

    # Run a query
    result = system.run(
        query="What is retrieval-augmented generation (RAG)?",
        session_id="demo-session-1",
    )

    # Print results
    print(f"\nQuery:      {result['query']}")
    print(f"Rewritten:  {result.get('rewritten_query')}")
    print(f"Intent:     {result.get('intent')}")
    print(f"Complexity: {result.get('complexity')}")
    print(f"Strategy:   {result.get('retrieval_strategy')}")
    print(f"Confidence: {result.get('confidence_score'):.2f}" if result.get("confidence_score") else "Confidence: N/A")
    print(f"\nAnswer:\n{result.get('answer')}")

    if result.get("evaluation_metrics"):
        print(f"\nEvaluation metrics: {result['evaluation_metrics']}")


# =============================================================================
# 2. Conversation memory — multi-turn dialogue
# =============================================================================

def demo_conversation_memory() -> None:
    """Demonstrate multi-turn conversation with persistent memory."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-turn Conversation Memory")
    print("=" * 60)

    from astra_rag import AstraRAGSystem
    from astra_rag.core.config import SystemConfig

    system = AstraRAGSystem(SystemConfig(), enable_self_improving=False)
    session = "demo-session-memory"

    turns = [
        "What is LangGraph?",
        "How does it compare to LangChain?",   # Relies on prior context
        "Can you give me a simple example?",   # Follow-up
    ]

    for i, query in enumerate(turns, 1):
        print(f"\n[Turn {i}] User: {query}")
        result = system.run(query, session_id=session)
        answer = result.get("answer") or "(no answer)"
        print(f"[Turn {i}] AI:   {answer[:300]}...")

    # Inspect memory
    from astra_rag.memory.conversation_memory import ConversationMemory

    memory = ConversationMemory(SystemConfig())
    history = memory.get_history(session)
    print(f"\nConversation history ({len(history)} messages stored)")


# =============================================================================
# 3. Adding documents to the knowledge base
# =============================================================================

def demo_knowledge_base() -> None:
    """Add documents to a vector store and retrieve them."""
    print("\n" + "=" * 60)
    print("DEMO 3: Knowledge Base (Vector Store)")
    print("=" * 60)

    from astra_rag.core.config import MemoryConfig, SystemConfig
    from astra_rag.memory.knowledge_memory import KnowledgeMemory

    config = SystemConfig(
        memory=MemoryConfig(
            vector_db_type="faiss",       # Use FAISS for local demo (no server)
            faiss_index_path="/tmp/demo_faiss",
        )
    )

    try:
        km = KnowledgeMemory(config)

        # Add some documents to the knowledge base
        documents = [
            {
                "content": (
                    "AstraRAG is an enterprise-grade Agentic RAG framework built on "
                    "LangGraph. It features multi-agent collaboration, self-improving "
                    "feedback loops, and support for multiple retrieval backends."
                ),
                "metadata": {"source": "docs", "topic": "AstraRAG"},
            },
            {
                "content": (
                    "LangGraph is a library for building stateful, multi-actor applications "
                    "with LLMs. It extends LangChain with a graph-based execution model "
                    "that supports cycles and conditional branching."
                ),
                "metadata": {"source": "docs", "topic": "LangGraph"},
            },
            {
                "content": (
                    "Retrieval-Augmented Generation (RAG) combines information retrieval "
                    "with language model generation. A retriever fetches relevant documents "
                    "and the LLM generates an answer grounded in those documents."
                ),
                "metadata": {"source": "docs", "topic": "RAG"},
            },
        ]

        ids = km.add_documents(documents)
        print(f"Added {len(ids)} documents to the knowledge base.")

        # Search the knowledge base
        results = km.search("What is AstraRAG?", top_k=2)
        print("\nSearch results for 'What is AstraRAG?':")
        for r in results:
            print(f"  Score={r['score']:.3f}: {r['content'][:100]}...")

        # Wire the knowledge base into the system
        from astra_rag import AstraRAGSystem

        system = AstraRAGSystem(config, knowledge_memory=km, enable_self_improving=False)
        result = system.run("Tell me about AstraRAG")
        print(f"\nAnswer with KB: {result.get('answer', '(no answer)')[:300]}")

    except Exception as exc:
        print(f"Knowledge base demo skipped: {exc}")
        print("(Ensure sentence-transformers or OpenAI API key is available)")


# =============================================================================
# 4. Async usage
# =============================================================================

async def demo_async() -> None:
    """Demonstrate async pipeline execution."""
    print("\n" + "=" * 60)
    print("DEMO 4: Async Execution")
    print("=" * 60)

    from astra_rag import AstraRAGSystem
    from astra_rag.core.config import SystemConfig

    system = AstraRAGSystem(SystemConfig(), enable_self_improving=False)

    # Run multiple queries concurrently
    queries = [
        ("What is vector embeddings?", "async-session-1"),
        ("Explain transformer architecture briefly.", "async-session-2"),
    ]

    tasks = [system.arun(q, session_id=sid) for q, sid in queries]
    results = await asyncio.gather(*tasks)

    for (q, _), result in zip(queries, results):
        print(f"\nQ: {q}")
        print(f"A: {(result.get('answer') or '(no answer)')[:200]}...")


# =============================================================================
# 5. Evaluation and self-improving metrics
# =============================================================================

def demo_evaluation() -> None:
    """Demonstrate the RAGAS-style evaluation layer."""
    print("\n" + "=" * 60)
    print("DEMO 5: Evaluation Layer")
    print("=" * 60)

    from astra_rag.core.config import SystemConfig
    from astra_rag.self_improving.evaluation import EvaluationLayer

    config = SystemConfig()
    evaluator = EvaluationLayer(config)

    # Example data
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    context = (
        "France is a country in Western Europe. Its capital and largest city "
        "is Paris, which is located in north-central France."
    )

    try:
        scores = evaluator.evaluate(
            question=question,
            answer=answer,
            context=context,
            metrics=["faithfulness", "answer_relevancy"],  # Subset for speed
        )
        print(f"\nEvaluation scores for: '{question}'")
        for metric, score in scores.items():
            print(f"  {metric:25s}: {score:.3f}")
    except Exception as exc:
        print(f"Evaluation demo skipped (requires OpenAI API key): {exc}")


# =============================================================================
# 6. Episodic memory — strategy recommendations
# =============================================================================

def demo_episodic_memory() -> None:
    """Show how episodic memory learns retrieval strategy preferences."""
    print("\n" + "=" * 60)
    print("DEMO 6: Episodic Memory")
    print("=" * 60)

    from astra_rag.memory.episodic_memory import EpisodicMemory

    mem = EpisodicMemory()

    # Simulate several interactions
    interactions = [
        ("factual", "simple", "vector", 0.9),
        ("factual", "simple", "vector", 0.85),
        ("factual", "simple", "graph", 0.6),
        ("analytical", "complex", "hybrid", 0.88),
        ("analytical", "complex", "vector", 0.55),
        ("analytical", "complex", "hybrid", 0.91),
    ]

    for intent, complexity, strategy, confidence in interactions:
        mem.record(
            session_id="demo",
            query=f"{intent} {complexity} query",
            intent=intent,
            complexity=complexity,
            strategy=strategy,
            confidence=confidence,
        )

    # Get strategy recommendations
    for intent, complexity in [("factual", "simple"), ("analytical", "complex")]:
        rec = mem.recommend_strategy(intent, complexity)
        print(f"  Recommended strategy for ({intent}, {complexity}): {rec}")

    print("\nPerformance trends:")
    for strategy, stats in mem.get_performance_trends().items():
        print(f"  {strategy}: {stats}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ASTRA_OPENAI_API_KEY")
    if not api_key:
        print(
            "\n⚠️  WARNING: OPENAI_API_KEY is not set.\n"
            "   Demos that call the LLM will fail.\n"
            "   Set it via:  export OPENAI_API_KEY=sk-...\n"
        )

    # Demos that don't require LLM API (always run)
    demo_episodic_memory()

    if api_key:
        # Demos that require a live OpenAI API key
        demo_basic_query()
        demo_conversation_memory()
        demo_knowledge_base()
        demo_evaluation()

        # Async demo
        asyncio.run(demo_async())
    else:
        print("\nSkipping LLM-dependent demos (no API key found).")
        print("Episodic memory demo completed successfully ✓")
