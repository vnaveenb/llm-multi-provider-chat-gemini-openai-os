"""FastAPI application entry point — Project 1: Multi-Provider Chat Core."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    project: str
    provider: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="Project 1 — Multi-Provider Chat Core",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check used by Docker HEALTHCHECK and CI smoke tests."""
    return HealthResponse(
        status="ok",
        project="project-1-multi-provider-chat",
        provider=os.environ.get("LLM_PROVIDER", "google_genai:gemini-2.0-flash"),
    )
