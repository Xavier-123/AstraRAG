"""
astra_rag/utils/llm.py
----------------------
Factory functions and utilities for LLM and embedding model access.

All LLM construction is centralised here so that changing the model or
provider requires editing a single file.

Key functions
-------------
get_llm(config)
    Returns a configured ``ChatOpenAI`` instance with tenacity retry.
get_embeddings(config)
    Returns an ``OpenAIEmbeddings`` (or sentence-transformer fallback).
count_tokens(text, model)
    Estimates token count without an API call using tiktoken.
StreamHandler
    Helper class that streams tokens to a callback for SSE / WebSocket use.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, List, Optional

import tiktoken
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from astra_rag.core.config import SystemConfig

logger = logging.getLogger(__name__)


# ── LLM factory ──────────────────────────────────────────────────────────────


def get_llm(config: SystemConfig, **overrides: Any) -> ChatOpenAI:
    """Build and return a ``ChatOpenAI`` instance from *config*.

    Parameters
    ----------
    config:
        Root system configuration.
    **overrides:
        Keyword arguments that override individual ``LLMConfig`` fields
        (e.g., ``temperature=0.7``).

    Returns
    -------
    ChatOpenAI
        Ready-to-invoke LangChain LLM wrapper.
    """
    llm_cfg = config.llm
    kwargs: dict[str, Any] = {
        "model": overrides.get("model", llm_cfg.model),
        "temperature": overrides.get("temperature", llm_cfg.temperature),
        "max_tokens": overrides.get("max_tokens", llm_cfg.max_tokens),
        "streaming": overrides.get("streaming", llm_cfg.streaming),
    }
    if llm_cfg.api_key:
        kwargs["api_key"] = llm_cfg.api_key
    if llm_cfg.base_url:
        kwargs["base_url"] = llm_cfg.base_url

    logger.debug("Creating ChatOpenAI(model=%s)", kwargs["model"])
    return ChatOpenAI(**kwargs)


# ── Embeddings factory ────────────────────────────────────────────────────────


def get_embeddings(config: SystemConfig) -> Any:
    """Return an embeddings model appropriate for the current configuration.

    Uses ``OpenAIEmbeddings`` by default.  Falls back gracefully to a
    sentence-transformers model when no API key is available (useful for
    local development / CI).

    Parameters
    ----------
    config:
        Root system configuration.
    """
    llm_cfg = config.llm
    if llm_cfg.api_key:
        kwargs: dict[str, Any] = {"model": "text-embedding-3-small"}
        if llm_cfg.api_key:
            kwargs["api_key"] = llm_cfg.api_key
        if llm_cfg.base_url:
            kwargs["base_url"] = llm_cfg.base_url
        return OpenAIEmbeddings(**kwargs)

    # Local fallback (no API key needed)
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.warning(
            "No OpenAI API key found – using local HuggingFace embeddings "
            "(sentence-transformers/all-MiniLM-L6-v2)."
        )
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        raise RuntimeError(
            "Cannot create embeddings: set OPENAI_API_KEY or install "
            "sentence-transformers (`pip install sentence-transformers`)."
        )


# ── Token counting ────────────────────────────────────────────────────────────


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return the approximate token count for *text* using tiktoken.

    Falls back to a character-based heuristic (~4 chars/token) when the
    tiktoken vocabulary file cannot be downloaded (e.g., in air-gapped
    environments or CI without internet access).

    Parameters
    ----------
    text:
        The string to measure.
    model:
        OpenAI model name – used to select the correct tokeniser.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except KeyError:
        pass
    except Exception:
        # Network error downloading vocab, or other tiktoken failure.
        # Fall through to heuristic.
        pass
    # Heuristic fallback: ~4 characters per token (slightly over-counts)
    return max(1, len(text) // 4)


def count_messages_tokens(messages: List[BaseMessage], model: str = "gpt-4o-mini") -> int:
    """Estimate total token count for a list of ``BaseMessage`` objects."""
    total = 0
    for m in messages:
        total += count_tokens(m.content if isinstance(m.content, str) else str(m.content), model)
    return total


def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    """Truncate *text* so that it fits within *max_tokens*.

    Uses tiktoken when available; falls back to character-based truncation.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except Exception:
        # Heuristic fallback: ~4 chars per token
        char_limit = max_tokens * 4
        return text[:char_limit]


# ── Retry-wrapped invoke helper ───────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    stop=stop_after_attempt(4),
    reraise=True,
)
def safe_invoke(llm: Any, messages: List[BaseMessage], **kwargs: Any) -> Any:
    """Invoke *llm* with automatic exponential-backoff retry.

    Useful when calling the LLM outside an agent context (e.g., from a
    utility script).
    """
    if kwargs:
        llm = llm.bind(**kwargs)
    return llm.invoke(messages)


# ── Streaming helper ──────────────────────────────────────────────────────────


class StreamCallbackHandler(BaseCallbackHandler):
    """LangChain callback that forwards streamed tokens to a caller-supplied
    function.

    Parameters
    ----------
    on_token:
        Called with each new token string as it is produced.

    Example::

        handler = StreamCallbackHandler(on_token=print)
        llm = get_llm(config, streaming=True, callbacks=[handler])
        llm.invoke([HumanMessage(content="Hello!")])
    """

    def __init__(self, on_token: Callable[[str], None]) -> None:
        super().__init__()
        self._on_token = on_token

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._on_token(token)


def stream_llm(
    config: SystemConfig,
    messages: List[BaseMessage],
    on_token: Optional[Callable[[str], None]] = None,
) -> Iterator[str]:
    """Stream tokens from the LLM, yielding each one.

    Parameters
    ----------
    config:
        System configuration.
    messages:
        Prompt messages.
    on_token:
        Optional side-effect callback called for each token.

    Yields
    ------
    str
        Individual token strings as they arrive.
    """
    llm = get_llm(config, streaming=True)
    for chunk in llm.stream(messages):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if on_token:
            on_token(token)
        yield token
