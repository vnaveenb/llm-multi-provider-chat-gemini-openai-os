"""FastAPI application entry point — Project 1: Multi-Provider Chat Core."""

from __future__ import annotations

import json
import logging
import os
import pathlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from shared.adapter import ProviderError
from shared.providers import get_embeddings, get_llm
from src.chat_core.pipeline import run_chat, run_tools, stream_chat
from src.chat_core.tools import get_tool_schemas

_UI = pathlib.Path(__file__).parent / "static" / "index.html"

logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    project: str
    provider: str


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: str = "You are a helpful AI assistant."


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    model: str
    tokens: TokenUsage


class StreamRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: str = "You are a helpful AI assistant."


class ToolsRequest(BaseModel):
    session_id: str
    message: str
    tools: list[str]
    system_prompt: str = "You are a helpful AI assistant."


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    name: str
    result: str


class ToolsResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    session_id: str
    model: str
    tokens: TokenUsage


class ReloadResponse(BaseModel):
    status: str
    provider: str


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(redis_url)
        await r.ping()
        await r.aclose()
        logger.info("Redis connected at %s", redis_url)
    except Exception:
        logger.warning("Redis unavailable at %s — conversation memory will use in-process fallback", redis_url)
    yield


app = FastAPI(
    title="Project 1 — Multi-Provider Chat Core",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_UI, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check used by Docker HEALTHCHECK and CI smoke tests."""
    return HealthResponse(
        status="ok",
        project="project-1-multi-provider-chat",
        provider=os.environ.get("LLM_PROVIDER", "google_genai:gemini-2.0-flash"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Synchronous chat — returns the full response once the LLM finishes."""
    try:
        result = await run_chat(request.session_id, request.message, request.system_prompt)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(
        answer=result["answer"],
        session_id=result["session_id"],
        model=result["model"],
        tokens=TokenUsage(**result["tokens"]),
    )


@app.post("/chat/stream")
async def chat_stream(request: StreamRequest) -> StreamingResponse:
    """Streaming chat — tokens arrive in real time via Server-Sent Events."""

    async def event_generator() -> AsyncIterator[str]:
        async for chunk in stream_chat(request.session_id, request.message, request.system_prompt):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/tools", response_model=ToolsResponse)
async def tools(request: ToolsRequest) -> ToolsResponse:
    """Function-calling chat — LLM may invoke registered server-side tools."""
    try:
        get_tool_schemas(request.tools)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await run_tools(
            request.session_id, request.message, request.tools, request.system_prompt
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ToolsResponse(
        answer=result["answer"],
        tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
        tool_results=[ToolResult(**tr) for tr in result["tool_results"]],
        session_id=result["session_id"],
        model=result["model"],
        tokens=TokenUsage(**result["tokens"]),
    )


@app.post("/reload-config", response_model=ReloadResponse)
async def reload_config() -> ReloadResponse:
    """Clear the LLM provider cache so the next request picks up updated env vars."""
    get_llm.cache_clear()
    get_embeddings.cache_clear()
    return ReloadResponse(
        status="reloaded",
        provider=os.environ.get("LLM_PROVIDER", "google_genai:gemini-2.0-flash"),
    )
