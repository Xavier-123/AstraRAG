"""
AstraRAG — Enterprise-grade Agentic RAG + Self-Improving RAG Framework
=======================================================================

AstraRAG combines LangGraph-orchestrated multi-agent collaboration with a
self-improving feedback loop to deliver reliable, production-ready
Retrieval-Augmented Generation.

Quick start::

    from astra_rag import AstraRAGSystem
    from astra_rag.core.config import SystemConfig

    config = SystemConfig()
    system = AstraRAGSystem(config)
    result = system.run("What is quantum entanglement?", session_id="demo")
    print(result["answer"])
"""

from astra_rag.workflow.graph import AstraRAGSystem

__version__ = "0.1.0"
__all__ = ["AstraRAGSystem", "__version__"]
