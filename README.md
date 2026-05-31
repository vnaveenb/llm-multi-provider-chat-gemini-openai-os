# Multi-Provider Chat Core

> A production-grade LLM chat API with streaming, conversation memory, function-calling, and automatic provider fallback — swap Gemini, GPT-4o, Claude, or Bedrock with a single environment variable and zero code changes.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20API-4285F4?logo=google&logoColor=white)
![Redis](https://img.shields.io/badge/Memory-Redis-DC382D?logo=redis&logoColor=white)
![LangSmith](https://img.shields.io/badge/Observability-LangSmith-FF6B35)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What is this?

Most LLM chat demos are hard-wired to one provider. The moment that provider raises prices, changes its API, or fails an availability SLA, you're rewriting code.

This project is built differently. The provider is a **runtime configuration value**, not a code dependency. The same FastAPI app, the same business logic, and the same test suite work identically against Gemini, GPT-4o, Claude, or Amazon Bedrock — switched by editing one environment variable. It demonstrates the full production chat stack: **streaming responses, Redis-backed conversation memory, function-calling with structured output, and automatic fallback** when a primary provider fails.

Built to show what "production-ready" actually means: not just working but measurable, resilient, and operationally manageable.

---

## Features

| Feature | Detail |
|---|---|
| Multi-provider routing | `LLM_PROVIDER=google_genai:gemini-2.0-flash` → swap to any provider, zero code change |
| Streaming | Server-Sent Events via FastAPI `StreamingResponse` — tokens arrive in real time |
| Conversation memory | Redis-backed `ConversationBufferWindowMemory` — stateful across requests |
| Function calling | LangChain tool-use with JSON schema binding |
| Structured output | Pydantic model binding via `.with_structured_output()` |
| Provider fallback | `with_fallbacks()` — automatic retry on secondary provider on failure |
| Token + cost tracking | Per-request token counts and estimated cost via LangSmith callbacks |
| Hot-swap providers | Change `LLM_PROVIDER` in env; next request uses the new model — no restart |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT / USER                           │
└──────────┬──────────────────┬───────────────────────────────────┘
           │                  │
    POST /chat           POST /chat/stream
           │                  │
           ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI  (src/chat_core/main.py)              │
│   /chat   /chat/stream   /tools   /health   /reload-config      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Chat Pipeline                                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Memory       │    │ Prompt       │    │ LLM Provider     │  │
│  │ (Redis)      │───▶│ Builder      │───▶│ (via providers.py│  │
│  │ Last N turns │    │ System +     │    │ + LangChain)     │  │
│  └──────────────┘    │ Context +    │    └────────┬─────────┘  │
│                       │ Question    │             │            │
│                       └──────────────┘             │            │
└──────────────────────────────────────────┬─────────┘            │
                                           │                       │
                    ┌──────────────────────┼───────────────────┐  │
                    │                      │                   │  │
              Gemini API            Vertex AI           OpenAI │  │
              Claude / Anthropic    Amazon Bedrock             │  │
              (primary)             (fallback)       (fallback)│  │
                    └──────────────────────────────────────────┘  │
```

### Flow 1 — Standard Chat

```
POST /chat  {session_id, message}
   │
   ▼  [memory.py]  Load last N turns from Redis
   │
   ▼  [pipeline.py]  Build prompt: system + history + user message
   │
   ▼  [providers.py]  get_llm() → cached provider instance
   │  LLM_PROVIDER env var determines which provider fires
   │
   ▼  LangChain invoke()  →  AIMessage
   │
   ▼  [memory.py]  Save turn to Redis
   │
   ✓  {"answer": "...", "session_id": "...", "model": "...", "tokens": {...}}
```

### Flow 2 — Streaming

```
POST /chat/stream  {session_id, message}
   │
   ▼  Same pipeline as Flow 1 up to the LLM call
   │
   ▼  LangChain astream()  →  AsyncIterator[AIMessageChunk]
   │
   ▼  FastAPI StreamingResponse  →  SSE: data: {"token": "..."}\n\n
   │  Tokens arrive at the client as they are generated
   │
   ✓  Stream closes with  data: {"done": true, "tokens": {...}}
```

### Flow 3 — Function Calling

```
POST /tools  {session_id, message, tools: [...]}
   │
   ▼  [pipeline.py]  Bind tools to model via .bind_tools(tools)
   │
   ▼  LLM decides whether to call a tool or respond directly
   │
   ▼  [adapter.py]  normalise_tool_calls()  →  [{name, args, id}]
   │
   ▼  Execute tool  →  ToolMessage
   │
   ▼  LLM resumes with tool result  →  final AIMessage
   │
   ✓  {"answer": "...", "tool_calls": [...], "tool_results": [...]}
```

### Flow 4 — Provider Fallback

```
Primary provider call fails (rate limit / outage)
   │
   ▼  LangChain with_fallbacks() catches the exception
   │
   ▼  Retries on configured fallback provider
   │  Primary: google_genai  →  Fallback: openai
   │
   ▼  [adapter.py]  check_safety_filter()  —  normalises finish_reason
   │
   ✓  Response delivered; failure logged to LangSmith
```

---

## Project Structure

```
├── .env.example               # Environment variable template
├── .github/workflows/ci.yml   # GitHub Actions: lint → typecheck → test → docker build
├── Dockerfile                 # Multi-stage python:3.11-slim, non-root user
├── requirements.txt           # All dependencies pinned
├── pyproject.toml             # ruff + mypy + pytest config
│
├── src/chat_core/
│   ├── main.py                # FastAPI app — /chat, /chat/stream, /tools, /health
│   ├── pipeline.py            # Chat orchestration: memory + prompt + LLM (to build)
│   ├── memory.py              # Redis conversation memory (to build)
│   └── routes/                # Endpoint handlers (to build)
│
├── shared/
│   ├── providers.py           # LLM + embeddings factory — one env var = full swap
│   ├── adapter.py             # Safety filter + tool-call normalisation
│   └── vector_schema.py       # Dimension-agnostic index config
│
└── tests/
    └── test_health.py         # Unit smoke tests — no API keys needed
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — at minimum, add your Gemini API key:

```env
GOOGLE_API_KEY=your-gemini-api-key
REDIS_URL=redis://localhost:6379
```

### 4. Start the server

```bash
uvicorn src.chat_core.main:app --reload --port 8001
```

Open [http://localhost:8001/docs](http://localhost:8001/docs) for the interactive Swagger UI.

---

## Demo Walkthrough

### Step 1 — Send a chat message

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-001", "message": "What is retrieval-augmented generation?"}'
```

### Step 2 — Streaming response (tokens arrive in real time)

```bash
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-001", "message": "Give me a summary in bullet points"}'
```

### Step 3 — Swap to a different provider (zero restart)

```bash
# Switch to OpenAI
export LLM_PROVIDER=openai:gpt-4o
export OPENAI_API_KEY=sk-...
uvicorn src.chat_core.main:app --port 8001

# Switch to Anthropic
export LLM_PROVIDER=anthropic:claude-opus-4-8
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn src.chat_core.main:app --port 8001
```

Same codebase. Same endpoints. Different intelligence. No code changes.

---

## Running Tests

```bash
# Unit tests — no API key, no network, runs in CI
pytest tests/ -m "not integration" -v

# Integration tests — requires live API key
pytest tests/ -v
```

---

## Docker

```bash
docker build -t multi-provider-chat:local .
docker run --env-file .env -p 8001:8001 multi-provider-chat:local
```

---

## Why This Matters for Enterprise

| Concern | How this project addresses it |
|---|---|
| **Vendor lock-in** | LangChain `init_chat_model()` factory abstracts all providers — switching is a config value, not a code change |
| **Resilience** | Automatic fallback to secondary provider on failure — uptime survives a provider outage |
| **Cost management** | Per-request token tracking; swap to cheaper provider by changing one env var |
| **Compliance** | Conversation memory stored in your own Redis instance — no chat history sent to third parties |
| **Observability** | LangSmith traces every call: latency, cost, token counts, tool calls — all visible in dashboards |

---

## Key Design Decisions

- **`init_chat_model()` not direct imports** — LangChain's provider-agnostic factory means adding a new provider requires only an env var, never a new `import` or `if/elif` block
- **`@lru_cache` on provider factory** — expensive provider initialisation (gRPC channel setup for Vertex, credential loading for Bedrock) happens once per process
- **`adapter.py` quarantine** — Gemini returns multi-part content blocks; OpenAI returns content filters differently; all quirks are normalised before business logic sees them
- **Redis for memory** — in-process dicts don't survive restarts or scale horizontally; Redis works across replicas and persists across deploys
- **`pytest -m "not integration"` in CI** — unit tests cover the FastAPI layer without requiring live API keys; integration tests run locally with real credentials

---

## License

MIT
