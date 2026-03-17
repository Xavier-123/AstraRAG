"""
astra_rag/agents/reasoning.py
------------------------------
ReasoningAgent — Pipeline position: node 7 of 8.

Responsibilities
----------------
* **Chain-of-Thought (CoT)**: For ``simple``/``medium`` queries the agent
  prompts the LLM to think step-by-step before answering.
* **Multi-hop reasoning**: For ``complex`` queries (``reasoning_mode =
  "multi_hop"``) the agent iterates: reason → identify knowledge gap →
  retrieve → reason again.  Each iteration appends a step to
  ``reasoning_chain``.
* **Tool use (extensible)**: A lightweight ``ToolRegistry`` allows callers to
  register Python callables (e.g. a calculator, a unit converter) that the
  LLM can invoke by name within a structured reasoning turn.

Iterative retrieval protocol
------------------------------
When the LLM response contains the special marker ``[NEED_MORE_INFO: <query>]``
the agent:
1. Stores the missing-information query.
2. Sets ``state["retrieval_strategy"] = "vector"`` and
   ``state["rewritten_query"] = <query>`` to trigger another retrieval pass.
3. Returns early so the LangGraph router can send control back to
   ``multi_retriever``.

The router in ``workflow/graph.py`` detects the presence of a new
``rewritten_query`` that differs from the original to decide whether to
loop back.

Output state keys
-----------------
* ``reasoning_chain``  – ordered list of reasoning step strings
* ``answer``           – final natural-language answer
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from astra_rag.core.base_agent import BaseAgent
from astra_rag.core.config import SystemConfig
from astra_rag.core.state import GraphState

logger = logging.getLogger(__name__)

_NEED_MORE_INFO_RE = re.compile(r"\[NEED_MORE_INFO:\s*(.+?)\]", re.DOTALL)
_MAX_HOPS = 3              # Safety limit for multi-hop iterations
_MAX_REASONING_LOG_CHARS = 500  # Max chars stored per hop in reasoning_chain log


# ── Tool registry ─────────────────────────────────────────────────────────────


class ToolRegistry:
    """Lightweight registry mapping tool names to Python callables.

    Example::

        registry = ToolRegistry()
        registry.register("calculator", lambda expr: str(eval(expr)))
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        self._tools[name] = fn

    def call(self, name: str, *args: Any, **kwargs: Any) -> str:
        if name not in self._tools:
            return f"[ERROR] Unknown tool: {name}"
        try:
            return str(self._tools[name](*args, **kwargs))
        except Exception as exc:
            return f"[ERROR] Tool '{name}' failed: {exc}"

    def describe(self) -> str:
        if not self._tools:
            return "No tools available."
        return "\n".join(f"- {name}" for name in self._tools)


# ── Agent ─────────────────────────────────────────────────────────────────────


class ReasoningAgent(BaseAgent):
    """Performs chain-of-thought and/or multi-hop reasoning to produce an answer.

    Parameters
    ----------
    config:
        System configuration.
    tool_registry:
        Optional tool registry for tool-use scenarios.
    """

    _COT_SYSTEM = (
        "You are an expert reasoning assistant for an enterprise knowledge system.\n"
        "Use the provided context to answer the question carefully and accurately.\n\n"
        "Instructions:\n"
        "1. Think step-by-step — label each step 'Step N:'.\n"
        "2. Base your answer ONLY on the provided context.\n"
        "3. If the context is insufficient, say so clearly.\n"
        "4. End with a clear, concise answer on a line starting 'Answer:'.\n"
        "{tool_section}"
    )

    _MULTIHOP_SYSTEM = (
        "You are an expert multi-hop reasoning assistant.\n"
        "You can request additional information by writing [NEED_MORE_INFO: <search query>].\n"
        "Use chain-of-thought reasoning. When you have enough information, "
        "provide the final answer on a line starting 'Answer:'.\n"
        "{tool_section}"
    )

    def __init__(
        self,
        config: SystemConfig,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        super().__init__(config)
        self.tools = tool_registry or ToolRegistry()

    @property
    def name(self) -> str:
        return "reasoning"

    @property
    def description(self) -> str:
        return (
            "Performs chain-of-thought and/or multi-hop reasoning over the "
            "engineered context to produce a well-grounded answer."
        )

    def run(self, state: GraphState) -> GraphState:
        self._log_entry(state)

        task_plan = state.get("task_plan") or {}
        reasoning_mode = task_plan.get("reasoning_mode", "chain_of_thought")
        query = state.get("rewritten_query") or state.get("query", "")
        context = state.get("context") or ""
        iteration = state.get("iteration_count", 0)

        chain: List[str] = list(state.get("reasoning_chain") or [])

        if reasoning_mode == "multi_hop":
            updates = self._multi_hop_reasoning(state, query, context, chain, iteration)
        else:
            updates = self._cot_reasoning(query, context, chain, reasoning_mode)

        self._log_exit(state)
        return self._update_state(state, updates)

    # ── CoT reasoning ─────────────────────────────────────────────────────

    def _cot_reasoning(
        self,
        query: str,
        context: str,
        chain: List[str],
        mode: str,
    ) -> Dict[str, Any]:
        tool_section = (
            f"\nAvailable tools:\n{self.tools.describe()}\n"
            "Call a tool by writing [TOOL: <name>(<arg>)].\n"
            if self.tools._tools
            else ""
        )
        system = self._COT_SYSTEM.format(tool_section=tool_section)
        context_block = f"Context:\n{context}\n\n" if context else ""
        user_content = f"{context_block}Question: {query}"

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ]

        try:
            response = self._call_llm(messages)
            raw = response.content
        except Exception as exc:
            logger.error("[reasoning] LLM call failed: %s", exc)
            return {
                "reasoning_chain": chain + [f"ERROR: {exc}"],
                "answer": "I was unable to generate an answer due to a system error.",
            }

        raw = self._handle_tool_calls(raw)
        chain.append(f"[CoT] {raw[:_MAX_REASONING_LOG_CHARS]}")
        answer = self._extract_answer(raw)
        return {"reasoning_chain": chain, "answer": answer}

    # ── Multi-hop reasoning ───────────────────────────────────────────────

    def _multi_hop_reasoning(
        self,
        state: GraphState,
        query: str,
        context: str,
        chain: List[str],
        iteration: int,
    ) -> Dict[str, Any]:
        if iteration >= _MAX_HOPS:
            logger.warning("[reasoning] Max hops reached (%d); forcing final answer.", _MAX_HOPS)
            return self._cot_reasoning(query, context, chain, "chain_of_thought")

        tool_section = (
            f"\nAvailable tools:\n{self.tools.describe()}\n"
            if self.tools._tools
            else ""
        )
        system = self._MULTIHOP_SYSTEM.format(tool_section=tool_section)
        context_block = f"Current context:\n{context}\n\n" if context else ""
        history = "\n".join(chain[-4:])  # Last 4 steps for context
        history_block = f"Reasoning so far:\n{history}\n\n" if history else ""
        user_content = f"{context_block}{history_block}Question: {query}"

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ]

        try:
            response = self._call_llm(messages)
            raw = response.content
        except Exception as exc:
            logger.error("[reasoning] Multi-hop LLM call failed: %s", exc)
            return {
                "reasoning_chain": chain + [f"ERROR: {exc}"],
                "answer": "I was unable to generate an answer due to a system error.",
            }

        raw = self._handle_tool_calls(raw)
        chain.append(f"[Hop {iteration + 1}] {raw[:_MAX_REASONING_LOG_CHARS]}")

        # Check if LLM requested more information
        more_info_match = _NEED_MORE_INFO_RE.search(raw)
        if more_info_match:
            follow_up_query = more_info_match.group(1).strip()
            logger.info("[reasoning] Multi-hop: requesting more info for '%s'", follow_up_query)
            return {
                "reasoning_chain": chain,
                "rewritten_query": follow_up_query,
                "iteration_count": iteration + 1,
                # answer intentionally omitted — will be filled after retrieval
            }

        answer = self._extract_answer(raw)
        return {"reasoning_chain": chain, "answer": answer}

    # ── Tool handling ─────────────────────────────────────────────────────

    def _handle_tool_calls(self, raw: str) -> str:
        """Detect and execute [TOOL: name(arg)] patterns inline."""
        tool_call_re = re.compile(r"\[TOOL:\s*(\w+)\(([^)]*)\)\]")
        result = raw
        for match in tool_call_re.finditer(raw):
            tool_name = match.group(1)
            tool_arg = match.group(2).strip("\"'")
            tool_result = self.tools.call(tool_name, tool_arg)
            result = result.replace(match.group(0), f"[TOOL_RESULT: {tool_result}]")
        return result

    # ── Answer extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(raw: str) -> str:
        """Extract the text after 'Answer:' or return the last paragraph."""
        lines = raw.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("answer:"):
                return "\n".join(lines[i:]).replace("Answer:", "").strip()
        # No explicit Answer: label — return full response
        return raw.strip()
