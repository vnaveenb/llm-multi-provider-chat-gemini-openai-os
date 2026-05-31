"""Chat pipeline — orchestrates memory, prompt, and LLM invocation."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from shared.adapter import check_safety_filter, extract_text, normalise_tool_calls
from shared.providers import get_llm
from src.chat_core.memory import get_history, get_windowed_messages
from src.chat_core.tools import execute_tool, get_tool_schemas

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def _build_llm() -> Any:
    """Return the primary LLM, wrapped with a fallback if FALLBACK_LLM_PROVIDER is set."""
    primary = get_llm()
    fallback_env = os.environ.get("FALLBACK_LLM_PROVIDER", "").strip()
    if fallback_env:
        try:
            if ":" in fallback_env:
                fb_provider, fb_model = fallback_env.split(":", 1)
            else:
                fb_provider, fb_model = fallback_env, None
            fallback = get_llm(provider_override=fb_provider, model_override=fb_model)
            return primary.with_fallbacks([fallback])
        except Exception as exc:
            logger.warning("Could not initialise fallback provider: %s", exc)
    return primary


def _extract_token_usage(message: AIMessage | AIMessageChunk) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "total": usage.get("total_tokens", 0),
    }


def _model_name(response: AIMessage) -> str:
    meta = getattr(response, "response_metadata", {}) or {}
    return meta.get("model_name") or os.environ.get("LLM_PROVIDER", "unknown")


async def run_chat(
    session_id: str,
    message: str,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Invoke the LLM, persist the turn, and return a structured response."""
    history = get_history(session_id)
    past = get_windowed_messages(session_id)

    llm = _build_llm()
    messages = [SystemMessage(content=system_prompt), *past, HumanMessage(content=message)]

    response: AIMessage = await llm.ainvoke(messages)
    check_safety_filter(response)

    answer = extract_text(response)
    await history.aadd_messages([HumanMessage(content=message), AIMessage(content=answer)])

    return {
        "answer": answer,
        "session_id": session_id,
        "model": _model_name(response),
        "tokens": _extract_token_usage(response),
    }


async def stream_chat(
    session_id: str,
    message: str,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> AsyncIterator[dict[str, Any]]:
    """Stream tokens from the LLM, persist the complete turn, yield SSE-friendly dicts."""
    history = get_history(session_id)
    past = get_windowed_messages(session_id)

    llm = _build_llm()
    messages = [SystemMessage(content=system_prompt), *past, HumanMessage(content=message)]

    full_text = ""
    last_chunk: AIMessageChunk | None = None

    async for chunk in llm.astream(messages):
        token = extract_text(chunk)
        full_text += token
        last_chunk = chunk
        yield {"token": token}

    await history.aadd_messages([HumanMessage(content=message), AIMessage(content=full_text)])

    tokens = _extract_token_usage(last_chunk) if last_chunk is not None else {"input": 0, "output": 0, "total": 0}
    yield {"done": True, "tokens": tokens}


async def run_tools(
    session_id: str,
    message: str,
    tool_names: list[str],
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Run a function-calling turn: bind tools, invoke, execute, re-invoke, return result."""
    history = get_history(session_id)
    past = get_windowed_messages(session_id)

    schemas = get_tool_schemas(tool_names)
    llm = get_llm()
    bound = llm.bind_tools(schemas)

    messages = [SystemMessage(content=system_prompt), *past, HumanMessage(content=message)]

    first: AIMessage = await bound.ainvoke(messages)
    check_safety_filter(first)

    tool_calls = normalise_tool_calls(first)
    tool_results: list[dict[str, str]] = []
    tool_messages: list[ToolMessage] = []

    for tc in tool_calls:
        result = execute_tool(tc["name"], tc["args"])
        tool_results.append({"name": tc["name"], "result": result})
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    if tool_messages:
        final: AIMessage = await bound.ainvoke([*messages, first, *tool_messages])
        check_safety_filter(final)
        answer = extract_text(final)
        tokens = _extract_token_usage(final)
    else:
        answer = extract_text(first)
        tokens = _extract_token_usage(first)

    await history.aadd_messages([HumanMessage(content=message), AIMessage(content=answer)])

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "session_id": session_id,
        "model": _model_name(first),
        "tokens": tokens,
    }


__all__ = ["run_chat", "stream_chat", "run_tools"]
