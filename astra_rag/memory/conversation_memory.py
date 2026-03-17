"""
astra_rag/memory/conversation_memory.py
-----------------------------------------
ConversationMemory — manages per-session conversation history.

Storage backends
----------------
``redis`` (primary)
    Uses Redis LPUSH / LRANGE with a TTL so old sessions are garbage-
    collected automatically.  Each message is JSON-serialised and stored
    under the key ``astra_rag:conv:{session_id}``.

``in-memory`` (fallback)
    A plain dict of ``{session_id: [BaseMessage, ...]}``.  Used when
    Redis is unavailable (e.g., local development, CI).

Public API
----------
``add_message(session_id, message)``
    Append a message to the session history.

``get_history(session_id) → List[BaseMessage]``
    Return up to ``max_history`` most recent messages.

``clear(session_id)``
    Delete all messages for a session.

``format_for_prompt(session_id) → List[BaseMessage]``
    Return LangChain message objects ready to prepend to a prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from astra_rag.core.config import SystemConfig

logger = logging.getLogger(__name__)

_SESSION_TTL = 60 * 60 * 24  # 24 hours


# ── Message serialisation ─────────────────────────────────────────────────────


def _serialise(message: BaseMessage) -> str:
    return json.dumps(
        {"type": type(message).__name__, "content": message.content}
    )


def _deserialise(raw: str) -> BaseMessage:
    data = json.loads(raw)
    msg_type = data["type"]
    content = data["content"]
    mapping = {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "SystemMessage": SystemMessage,
    }
    cls = mapping.get(msg_type, HumanMessage)
    return cls(content=content)


# ── Backends ──────────────────────────────────────────────────────────────────


class _InMemoryBackend:
    def __init__(self, max_history: int) -> None:
        self._store: Dict[str, List[str]] = {}
        self._max = max_history

    def push(self, session_id: str, serialised: str) -> None:
        msgs = self._store.setdefault(session_id, [])
        msgs.append(serialised)
        # Trim to window
        if len(msgs) > self._max:
            self._store[session_id] = msgs[-self._max :]

    def get(self, session_id: str, count: int) -> List[str]:
        return self._store.get(session_id, [])[-count:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class _RedisBackend:
    def __init__(self, redis_url: str, max_history: int) -> None:
        import redis

        self._client = redis.from_url(redis_url, decode_responses=True)
        self._max = max_history

    def _key(self, session_id: str) -> str:
        return f"astra_rag:conv:{session_id}"

    def push(self, session_id: str, serialised: str) -> None:
        key = self._key(session_id)
        pipe = self._client.pipeline()
        pipe.rpush(key, serialised)
        pipe.ltrim(key, -self._max, -1)
        pipe.expire(key, _SESSION_TTL)
        pipe.execute()

    def get(self, session_id: str, count: int) -> List[str]:
        key = self._key(session_id)
        return self._client.lrange(key, -count, -1)

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))


# ── Public class ──────────────────────────────────────────────────────────────


class ConversationMemory:
    """Stores and retrieves per-session conversation history.

    Parameters
    ----------
    config:
        System configuration.  Uses ``memory.redis_url`` and
        ``memory.max_history``.
    """

    def __init__(self, config: SystemConfig) -> None:
        self._max = config.memory.max_history
        self._backend = self._init_backend(config)

    @staticmethod
    def _init_backend(config: SystemConfig) -> Any:
        try:
            backend = _RedisBackend(config.memory.redis_url, config.memory.max_history)
            # Quick connectivity check
            backend._client.ping()
            logger.info("[ConversationMemory] Using Redis backend.")
            return backend
        except Exception as exc:
            logger.warning(
                "[ConversationMemory] Redis unavailable (%s); using in-memory backend.", exc
            )
            return _InMemoryBackend(config.memory.max_history)

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Append *message* to the history for *session_id*."""
        self._backend.push(session_id, _serialise(message))

    def get_history(
        self, session_id: str, max_turns: Optional[int] = None
    ) -> List[BaseMessage]:
        """Return the most recent messages for *session_id*.

        Parameters
        ----------
        session_id:
            Session identifier.
        max_turns:
            If provided, return at most this many *turns* (1 turn = 1 human
            + 1 AI message, so ``max_turns * 2`` raw messages).
        """
        count = (max_turns * 2) if max_turns else self._max * 2
        raw = self._backend.get(session_id, count)
        messages: List[BaseMessage] = []
        for r in raw:
            try:
                messages.append(_deserialise(r))
            except Exception as exc:
                logger.warning("[ConversationMemory] Failed to deserialise message: %s", exc)
        return messages

    def clear(self, session_id: str) -> None:
        """Delete all stored messages for *session_id*."""
        self._backend.clear(session_id)

    def format_for_prompt(
        self, session_id: str, max_turns: int = 5
    ) -> List[BaseMessage]:
        """Return conversation history formatted as LangChain messages.

        Suitable for prepending to a prompt's message list.
        """
        return self.get_history(session_id, max_turns=max_turns)
