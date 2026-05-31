# Project 1 — Multi-Provider Chat Core

Provider-agnostic LLM chat API built on LangChain. Demonstrates streaming responses, conversation memory, function-calling, structured output, and provider routing/fallback — all without changing a line of business logic.

**Target roles:** V Group, Trianz, TCS, Fuzen, DNB Sr. Analyst

## What it covers

| Capability | Implementation |
|---|---|
| Multi-provider routing | `LLM_PROVIDER` env var via `shared/providers.py` → Gemini, Vertex, OpenAI, Anthropic, Bedrock |
| Streaming | Server-Sent Events via FastAPI `StreamingResponse` |
| Conversation memory | Redis-backed `ConversationBufferWindowMemory` |
| Function calling | LangChain tool-use with structured input/output |
| Structured output | Pydantic model binding via `model.with_structured_output()` |
| Provider fallback | LangChain `with_fallbacks()` for resilience |
| Token + cost tracking | LangSmith callback integration |

## Quick start

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
pip install -r requirements.txt
uvicorn src.chat_core.main:app --reload --port 8001
# → http://localhost:8001/health
# → http://localhost:8001/docs
```

## Docker

```bash
docker build -t multi-provider-chat:local .
docker run --env-file .env -p 8001:8001 multi-provider-chat:local
```

## Swap provider (zero code changes)

```bash
# Switch to Vertex AI
LLM_PROVIDER=google_vertexai:gemini-2.0-flash-001 uvicorn src.chat_core.main:app --port 8001

# Switch to OpenAI
LLM_PROVIDER=openai:gpt-4o OPENAI_API_KEY=sk-... uvicorn src.chat_core.main:app --port 8001
```

## CI pipeline

GitHub Actions runs on every push: **lint (ruff) → type-check (mypy) → unit tests (pytest) → Docker build**.
The Docker image is built but not pushed until you configure registry credentials.

## Architecture

```
src/chat_core/
├── main.py          FastAPI app + /health endpoint
├── routes/          Chat, streaming, tool-use endpoints (to be built)
└── memory.py        Redis conversation memory (to be built)

shared/
├── providers.py     LLM + embeddings factory (one env var = full provider swap)
├── adapter.py       Safety filter + tool-call normalisation
└── vector_schema.py Dimension-agnostic vector index config
```
