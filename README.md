# Multi-Provider Chat Core

> A production-grade LLM chat API with real-time streaming, Redis-backed conversation memory, function calling, and automatic provider fallback — swap Gemini, GPT-4o, Claude, or Bedrock with a single environment variable and zero code changes. Ships with a built-in dark-theme chat UI.

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

This project is built differently. The provider is a **runtime configuration value**, not a code dependency. The same FastAPI app, the same business logic, and the same test suite work identically against Gemini, GPT-4o, Claude, or Amazon Bedrock — switched by editing one environment variable. It demonstrates the full production chat stack: **real-time streaming, Redis-backed conversation memory, function calling, and automatic provider fallback**.

---

## Features

| Feature | Detail |
|---|---|
| **Multi-provider routing** | `LLM_PROVIDER=google_genai:gemini-2.0-flash` → swap any provider, zero code change |
| **Real-time streaming** | Server-Sent Events via FastAPI `StreamingResponse` — tokens appear as they are generated |
| **Conversation memory** | Redis-backed history, windowed to last 10 turns — stateful across requests |
| **Function calling** | Server-side tool registry; LLM decides which tools to invoke |
| **Provider fallback** | `with_fallbacks()` — automatic retry on a secondary provider when the primary fails |
| **Token tracking** | Per-request input/output counts via `usage_metadata` |
| **Hot-swap providers** | `POST /reload-config` clears the LRU cache; next request picks up the new `LLM_PROVIDER` env var |
| **Built-in chat UI** | Served at `/` — three-mode interface (Chat · Stream · Tools) with blinking-cursor streaming |

---

## Quick start

### With Docker Compose (recommended)

```bash
# 1. Copy and fill in your API key
cp .env.example .env
# edit .env — set GOOGLE_API_KEY=<your-key>

# 2. Start the stack (app + Redis)
docker compose up --build
```

Open **http://localhost:8001** for the chat UI, or **http://localhost:8001/docs** for the Swagger API explorer.

### Without Docker (local dev)

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # add your GOOGLE_API_KEY

uvicorn src.chat_core.main:app --reload --port 8001
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Browser  /  curl  /  API client                │
└───────┬──────────────────┬──────────────────────────────────────┘
        │                  │
   GET /            POST /chat
   (Chat UI)        POST /chat/stream
                    POST /tools
                    POST /reload-config
                    GET  /health
        │                  │
        ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              FastAPI  (src/chat_core/main.py)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         memory.py    pipeline.py   tools.py
         (Redis /     (orchestrate  (tool registry:
         in-memory    LLM calls)    get_current_time,
         fallback)                  calculate)
              │            │
              └────────────┴──────────▶ shared/providers.py
                                        (get_llm via init_chat_model)
                                              │
                           ┌──────────────────┼──────────────────┐
                           ▼                  ▼                  ▼
                      Gemini API         Vertex AI          OpenAI / Claude /
                      (primary)          (swap-ready)       Bedrock (fallback)
```

### Flow 1 — Chat (sync)

```
POST /chat  { session_id, message }
  ↓  load last 10 turns from Redis
  ↓  build [SystemMessage, ...history, HumanMessage]
  ↓  llm.ainvoke()  →  AIMessage
  ↓  save turn to Redis
  →  { answer, session_id, model, tokens: { input, output, total } }
```

### Flow 2 — Streaming

```
POST /chat/stream  { session_id, message }
  ↓  same pipeline setup
  ↓  llm.astream()  →  AsyncIterator[AIMessageChunk]
  ↓  FastAPI StreamingResponse  →  SSE: data: {"token":"…"}\n\n
     (each token flushed immediately as it arrives)
  ↓  stream closes with   data: {"done":true,"tokens":{…}}\n\n
  ↓  complete turn saved to Redis
```

### Flow 3 — Function calling

```
POST /tools  { session_id, message, tools: ["get_current_time","calculate"] }
  ↓  get_tool_schemas(tools)  →  bind to llm.bind_tools()
  ↓  first ainvoke()  →  AIMessage with tool_calls
  ↓  execute_tool(name, args)  →  string result
  ↓  second ainvoke() with ToolMessages  →  final AIMessage
  →  { answer, tool_calls, tool_results, session_id, model, tokens }
```

### Flow 4 — Provider fallback

```
Primary provider call fails (rate limit / outage)
  ↓  LangChain with_fallbacks() catches the exception
  ↓  retries on FALLBACK_LLM_PROVIDER
  ↓  adapter.py normalises the response across providers
  →  response delivered transparently to the caller
```

---

## Project structure

```
├── .env.example                   # Environment variable template
├── .github/workflows/ci.yml       # GitHub Actions: lint → typecheck → test → docker build
├── docker-compose.yml             # App + Redis; one command to run everything
├── Dockerfile                     # Multi-stage python:3.11-slim, non-root user
├── requirements.txt               # Pinned dependencies
├── pyproject.toml                 # ruff + mypy + pytest config
│
├── src/chat_core/
│   ├── main.py                    # FastAPI app — all endpoints + UI route
│   ├── pipeline.py                # Chat orchestration: memory + prompt + LLM
│   ├── memory.py                  # Redis history (in-process fallback if Redis is down)
│   ├── tools.py                   # Tool registry: get_current_time, calculate
│   └── static/
│       └── index.html             # Built-in chat UI (Chat · Stream · Tools tabs)
│
├── shared/
│   ├── providers.py               # LLM + embeddings factory — one env var = full swap
│   ├── adapter.py                 # Safety filter + tool-call normalisation
│   └── vector_schema.py           # Dimension-agnostic index config (used by later projects)
│
└── tests/
    ├── test_health.py             # /health smoke tests
    └── test_chat.py               # Endpoint + calculator unit tests (no API key needed)
```

---

## API reference

### `GET /`
Serves the built-in chat UI.

---

### `POST /chat`
Synchronous chat — waits for the full LLM response.

**Request**
```json
{ "session_id": "demo-001", "message": "What is retrieval-augmented generation?", "system_prompt": "You are a helpful assistant." }
```

**Response**
```json
{
  "answer": "RAG is a technique that ...",
  "session_id": "demo-001",
  "model": "gemini-2.0-flash",
  "tokens": { "input": 42, "output": 118, "total": 160 }
}
```

---

### `POST /chat/stream`
Streaming chat — tokens arrive via Server-Sent Events.

**Request** — same shape as `/chat`.

**SSE stream**
```
data: {"token":"RAG"}
data: {"token":" stands"}
data: {"token":" for"}
...
data: {"done":true,"tokens":{"input":42,"output":118,"total":160}}
```

**JavaScript client**
```js
const res = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ session_id: 'demo-001', message: 'What is RAG?' }),
});
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = '';
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buf += dec.decode(value, { stream: true });
  for (const part of buf.split('\n\n')) {
    if (part.startsWith('data: ')) {
      const ev = JSON.parse(part.slice(6));
      if (ev.token) process.stdout.write(ev.token);
    }
  }
}
```

---

### `POST /tools`
Function-calling chat — the LLM can invoke registered server-side tools.

**Available tools**

| Tool | Description | Args |
|---|---|---|
| `get_current_time` | Returns current UTC time (ISO 8601) | none |
| `calculate` | Safe arithmetic evaluator (+, -, *, /, ^, mod) | `expression: str` |

**Request**
```json
{
  "session_id": "demo-001",
  "message": "What time is it, and what is 2^10?",
  "tools": ["get_current_time", "calculate"]
}
```

**Response**
```json
{
  "answer": "The current time is 2026-05-31T10:22:00Z and 2^10 equals 1024.",
  "tool_calls": [
    { "id": "tc-1", "name": "get_current_time", "args": {} },
    { "id": "tc-2", "name": "calculate", "args": { "expression": "2^10" } }
  ],
  "tool_results": [
    { "name": "get_current_time", "result": "2026-05-31T10:22:00.000000+00:00" },
    { "name": "calculate", "result": "1024" }
  ],
  "session_id": "demo-001",
  "model": "gemini-2.0-flash",
  "tokens": { "input": 88, "output": 52, "total": 140 }
}
```

---

### `POST /reload-config`
Clears the LRU-cached LLM instance. Update `LLM_PROVIDER` in the environment, call this endpoint, and the next request uses the new provider — no restart required.

**Response**
```json
{ "status": "reloaded", "provider": "openai:gpt-4o" }
```

---

### `GET /health`
Health check used by Docker and CI.

**Response**
```json
{ "status": "ok", "project": "project-1-multi-provider-chat", "provider": "google_genai:gemini-2.0-flash" }
```

---

## curl demo

```bash
# Standard chat
curl -s -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","message":"What is RAG?"}' | python -m json.tool

# Streaming (tokens print as they arrive)
curl -s -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","message":"Summarise in 3 bullets"}' \
  --no-buffer

# Function calling
curl -s -X POST http://localhost:8001/tools \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","message":"What time is it and what is 2^10?","tools":["get_current_time","calculate"]}' \
  | python -m json.tool

# Hot-swap provider (no restart)
export LLM_PROVIDER=openai:gpt-4o
export OPENAI_API_KEY=sk-...
curl -s -X POST http://localhost:8001/reload-config | python -m json.tool
```

---

## Provider swap

Change one variable — everything else stays identical:

```env
# Gemini API (default)
LLM_PROVIDER=google_genai:gemini-2.0-flash
GOOGLE_API_KEY=...

# Vertex AI
LLM_PROVIDER=google_vertexai:gemini-2.0-flash-001
GOOGLE_CLOUD_PROJECT=...

# OpenAI
LLM_PROVIDER=openai:gpt-4o
OPENAI_API_KEY=...

# Anthropic
LLM_PROVIDER=anthropic:claude-opus-4-8
ANTHROPIC_API_KEY=...

# AWS Bedrock
LLM_PROVIDER=bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

For automatic fallback, set a secondary provider:
```env
FALLBACK_LLM_PROVIDER=openai:gpt-4o
```

---

## Running tests

```bash
# Unit tests — no API key, no network, runs in CI
pytest tests/ -m "not integration" -v

# All tests (requires live API key)
pytest tests/ -v
```

---

## Why this matters for enterprise

| Concern | How this project addresses it |
|---|---|
| **Vendor lock-in** | `init_chat_model()` factory — adding a new provider is a config change, never a code change |
| **Resilience** | Automatic fallback to a secondary provider on failure — uptime survives a provider outage |
| **Cost management** | Per-request token tracking; swap to a cheaper model by changing one env var |
| **Compliance** | Conversation history stored in your own Redis — no chat history sent to third parties |
| **Observability** | LangSmith traces every call: latency, cost, token counts, tool calls |
| **Operability** | Hot-swap providers without restarting the service via `POST /reload-config` |

---

## Key design decisions

- **`init_chat_model()` not direct imports** — adding a new LLM provider requires only an env var, never a new `import` or `if/elif` chain
- **`@lru_cache` on provider factory** — expensive initialisation (gRPC channels for Vertex, credential loading for Bedrock) happens once per process
- **`adapter.py` quarantine** — Gemini returns multi-part content blocks; OpenAI uses different `finish_reason` strings; all quirks are normalised before business logic sees them
- **Redis for memory** — in-process dicts don't survive restarts or scale horizontally; Redis works across replicas and persists across deploys; silent in-process fallback for local dev without Redis
- **Safe AST evaluator** — the `calculate` tool uses Python's `ast` module to walk the expression tree rather than `eval()`, so it cannot execute arbitrary code
- **`pytest -m "not integration"` in CI** — unit tests cover all endpoints with mocked LLMs; no API key required in CI

---

## License

MIT
