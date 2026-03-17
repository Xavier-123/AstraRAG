"""
astra_rag/core/base_agent.py
-----------------------------
Abstract base class that every AstraRAG agent inherits from.

Design principles
-----------------
* **Single responsibility** – each subclass owns one well-defined pipeline step.
* **Decoupling** – agents communicate exclusively through ``GraphState``; no
  direct agent-to-agent calls.
* **Resilience** – every LLM call is wrapped in a tenacity retry loop with
  exponential back-off so transient API errors don't fail a user query.
* **Observability** – structured logging is wired in at this level so every
  agent emits consistent log lines without extra boilerplate.

Sub-classing contract
---------------------
Implement the three abstract members::

    class MyAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "my_agent"

        @property
        def description(self) -> str:
            return "Does something useful."

        def run(self, state: GraphState) -> GraphState:
            # … agent logic …
            return self._update_state(state, {"answer": "hello"})
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState


class BaseAgent(ABC):
    """Abstract base for all AstraRAG agents.

    Parameters
    ----------
    config:
        System-wide configuration injected at construction time.
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self._logger = logging.getLogger(f"astra_rag.agents.{self.name}")
        if config.debug:
            self._logger.setLevel(logging.DEBUG)
        self._llm: Optional[Any] = None  # lazily initialised

    # ── Abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique snake_case identifier used as the LangGraph node name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description of what this agent does."""

    @abstractmethod
    def run(self, state: GraphState) -> GraphState:
        """Execute the agent's logic and return the updated state.

        Parameters
        ----------
        state:
            Current shared pipeline state.

        Returns
        -------
        GraphState
            A *partial* dict containing only the keys this agent modifies.
            LangGraph will merge it into the canonical state.
        """

    # ── Async wrapper ─────────────────────────────────────────────────────

    async def arun(self, state: GraphState) -> GraphState:
        """Async-compatible wrapper around :meth:`run`.

        Agents that have genuine async I/O should override *this* method
        directly and call ``await`` on async sub-routines.  The default
        implementation offloads the synchronous ``run`` to a thread pool so
        the event loop is not blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, state)

    # ── LLM helpers ───────────────────────────────────────────────────────

    def _get_llm(self) -> Any:
        """Lazily initialise and cache the ChatOpenAI instance."""
        if self._llm is None:
            from astra_rag.utils.llm import get_llm

            self._llm = get_llm(self.config)
        return self._llm

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_llm(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        """Call the LLM with automatic retry on transient failures.

        Parameters
        ----------
        messages:
            Conversation/prompt messages in LangChain format.
        **kwargs:
            Additional parameters forwarded to the LLM (e.g., ``temperature``).

        Returns
        -------
        AIMessage
            Raw LLM response message.
        """
        llm = self._get_llm()
        self._logger.debug("Calling LLM with %d messages", len(messages))
        if kwargs:
            llm = llm.bind(**kwargs)
        return llm.invoke(messages)

    async def _acall_llm(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        """Async variant of :meth:`_call_llm`."""
        llm = self._get_llm()
        self._logger.debug("Async calling LLM with %d messages", len(messages))
        if kwargs:
            llm = llm.bind(**kwargs)
        return await llm.ainvoke(messages)

    # ── State helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _update_state(state: GraphState, updates: Dict[str, Any]) -> GraphState:
        """Return a new state dict with *updates* merged in.

        LangGraph accepts partial dicts from nodes, so we can simply return
        the updates dict.  This helper is provided for clarity and to allow
        agents to build on the existing state when needed.

        Parameters
        ----------
        state:
            Existing state (used as base when building full dicts).
        updates:
            Keys to set / overwrite.
        """
        return {**state, **updates}  # type: ignore[return-value]

    # ── Logging convenience ───────────────────────────────────────────────

    def _log_entry(self, state: GraphState) -> None:
        self._logger.info(
            "[%s] ▶ entering | query='%s' | iter=%d",
            self.name,
            (state.get("query") or "")[:80],
            state.get("iteration_count", 0),
        )

    def _log_exit(self, state: GraphState) -> None:
        self._logger.info("[%s] ◀ done", self.name)
