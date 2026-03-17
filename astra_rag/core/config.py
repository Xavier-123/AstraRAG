"""
astra_rag/core/config.py
------------------------
Central configuration model for the AstraRAG system.

All settings can be overridden via environment variables (prefixed with
``ASTRA_``).  The ``SystemConfig`` class uses Pydantic v2's
``model_validator`` to load values from the environment at construction
time, so simply setting ``ASTRA_LLM_MODEL=gpt-4o`` before instantiation is
enough to change the model.

Environment variable mapping
-----------------------------
``ASTRA_LLM_MODEL``               → ``llm.model``
``ASTRA_LLM_TEMPERATURE``         → ``llm.temperature``
``ASTRA_LLM_MAX_TOKENS``          → ``llm.max_tokens``
``ASTRA_OPENAI_API_KEY``          → ``llm.api_key``
``ASTRA_OPENAI_BASE_URL``         → ``llm.base_url``
``ASTRA_RETRIEVAL_TOP_K``         → ``retrieval.top_k``
``ASTRA_RETRIEVAL_THRESHOLD``     → ``retrieval.similarity_threshold``
``ASTRA_REDIS_URL``               → ``memory.redis_url``
``ASTRA_VECTOR_DB``               → ``memory.vector_db_type``
``ASTRA_MAX_HISTORY``             → ``memory.max_history``
``ASTRA_EVAL_INTERVAL``           → ``self_improving.evaluation_interval``
``ASTRA_OPT_THRESHOLD``           → ``self_improving.optimization_threshold``
``ASTRA_MAX_ITERATIONS``          → ``max_reflection_iterations``
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Load .env file if present (no-op when running in CI with real env vars)
load_dotenv()


class LLMConfig(BaseModel):
    """Settings for the primary language model."""

    model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    base_url: Optional[str] = Field(default=None, description="Custom OpenAI-compatible base URL")
    streaming: bool = False

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict) -> dict:
        values.setdefault("model", os.getenv("ASTRA_LLM_MODEL", "gpt-4o-mini"))
        values.setdefault("temperature", float(os.getenv("ASTRA_LLM_TEMPERATURE", "0.0")))
        values.setdefault("max_tokens", int(os.getenv("ASTRA_LLM_MAX_TOKENS", "2048")))
        values.setdefault("api_key", os.getenv("ASTRA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))
        values.setdefault("base_url", os.getenv("ASTRA_OPENAI_BASE_URL"))
        return values


class RetrievalConfig(BaseModel):
    """Settings controlling document retrieval behaviour."""

    top_k: int = Field(default=5, gt=0, description="Number of documents to retrieve")
    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum similarity score to keep"
    )
    rerank_top_k: int = Field(default=3, gt=0, description="Documents kept after reranking")
    max_context_tokens: int = Field(default=4096, description="Token budget for assembled context")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict) -> dict:
        values.setdefault("top_k", int(os.getenv("ASTRA_RETRIEVAL_TOP_K", "5")))
        values.setdefault(
            "similarity_threshold", float(os.getenv("ASTRA_RETRIEVAL_THRESHOLD", "0.5"))
        )
        return values


class MemoryConfig(BaseModel):
    """Settings for the memory subsystems."""

    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    vector_db_type: str = Field(
        default="chroma", description="Vector DB backend: chroma | faiss | milvus | weaviate"
    )
    max_history: int = Field(default=20, gt=0, description="Max conversation turns to keep")
    chroma_persist_dir: str = Field(default="./chroma_db")
    faiss_index_path: str = Field(default="./faiss_index")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict) -> dict:
        values.setdefault("redis_url", os.getenv("ASTRA_REDIS_URL", "redis://localhost:6379/0"))
        values.setdefault("vector_db_type", os.getenv("ASTRA_VECTOR_DB", "chroma"))
        values.setdefault("max_history", int(os.getenv("ASTRA_MAX_HISTORY", "20")))
        return values


class SelfImprovingConfig(BaseModel):
    """Settings for the self-improving feedback loop."""

    evaluation_interval: int = Field(
        default=10, gt=0, description="Evaluate every N interactions"
    )
    optimization_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Average score below which optimisation is triggered",
    )
    enable_prompt_optimization: bool = True
    enable_retriever_tuning: bool = True
    max_optimization_rounds: int = Field(default=3, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict) -> dict:
        values.setdefault(
            "evaluation_interval", int(os.getenv("ASTRA_EVAL_INTERVAL", "10"))
        )
        values.setdefault(
            "optimization_threshold", float(os.getenv("ASTRA_OPT_THRESHOLD", "0.7"))
        )
        return values


class SystemConfig(BaseModel):
    """Root configuration object passed throughout the AstraRAG system.

    Example::

        config = SystemConfig(
            llm=LLMConfig(model="gpt-4o"),
            retrieval=RetrievalConfig(top_k=8),
        )
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    self_improving: SelfImprovingConfig = Field(default_factory=SelfImprovingConfig)
    max_reflection_iterations: int = Field(
        default=3, gt=0, description="Max answer-regeneration cycles per query"
    )
    debug: bool = Field(default=False, description="Enable verbose debug logging")

    @model_validator(mode="before")
    @classmethod
    def _load_from_env(cls, values: dict) -> dict:
        values.setdefault(
            "max_reflection_iterations",
            int(os.getenv("ASTRA_MAX_ITERATIONS", "3")),
        )
        values.setdefault("debug", os.getenv("ASTRA_DEBUG", "false").lower() == "true")
        return values
