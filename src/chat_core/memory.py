"""Redis-backed conversation memory with in-process fallback."""

from __future__ import annotations

import logging
import os

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

_IN_MEMORY_STORE: dict[str, InMemoryChatMessageHistory] = {}

_REDIS_TTL = 3600


def get_history(session_id: str) -> BaseChatMessageHistory:
    """Return a message history store for the given session.

    Tries Redis first; silently falls back to in-process memory if Redis
    is unavailable or the connection fails.
    """
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory

        store = RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url,
            ttl=_REDIS_TTL,
        )
        store.get_messages()
        return store
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — using in-process store for %r", exc, session_id)

    if session_id not in _IN_MEMORY_STORE:
        _IN_MEMORY_STORE[session_id] = InMemoryChatMessageHistory()
    return _IN_MEMORY_STORE[session_id]


def get_windowed_messages(session_id: str, max_turns: int = 10) -> list[BaseMessage]:
    """Return the last *max_turns* human+AI pairs from the session history."""
    messages = get_history(session_id).messages
    return messages[-(max_turns * 2):]


__all__ = ["get_history", "get_windowed_messages"]
