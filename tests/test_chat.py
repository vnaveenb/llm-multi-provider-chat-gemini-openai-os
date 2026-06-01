"""Unit tests for chat endpoints — no LLM calls, no network required."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.chat_core.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _mock_ai_message(text: str = "test response", tool_calls: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    msg.response_metadata = {"model_name": "mock-model"}
    msg.tool_calls = tool_calls or []
    return msg


def _make_mock_llm(answer: str = "test response") -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=_mock_ai_message(answer))
    llm.with_fallbacks = MagicMock(return_value=llm)
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


# ── /chat ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_chat_returns_200(client: TestClient) -> None:
    with patch("src.chat_core.pipeline.get_llm", return_value=_make_mock_llm()):
        response = client.post("/chat", json={"session_id": "t-001", "message": "Hello"})
    assert response.status_code == 200


@pytest.mark.unit
def test_chat_response_schema(client: TestClient) -> None:
    with patch("src.chat_core.pipeline.get_llm", return_value=_make_mock_llm("hi there")):
        data = client.post("/chat", json={"session_id": "t-002", "message": "Hi"}).json()
    assert data["answer"] == "hi there"
    assert data["session_id"] == "t-002"
    assert "model" in data
    assert set(data["tokens"].keys()) == {"input", "output", "total", "cached"}


# ── /chat/stream ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_chat_stream_sse_format(client: TestClient) -> None:
    async def _fake_stream(*_args: object, **_kwargs: object):
        yield {"token": "Hello"}
        yield {"token": " world"}
        yield {"done": True, "tokens": {"input": 5, "output": 2, "total": 7}}

    with patch("src.chat_core.main.stream_chat", side_effect=_fake_stream):
        response = client.post(
            "/chat/stream", json={"session_id": "t-003", "message": "Hi"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [e for e in response.text.split("\n\n") if e.strip()]
    assert all(e.startswith("data: ") for e in events)
    last = json.loads(events[-1][len("data: "):])
    assert last.get("done") is True


@pytest.mark.unit
def test_chat_stream_yields_tokens(client: TestClient) -> None:
    async def _fake_stream(*_args: object, **_kwargs: object):
        for word in ["one", " two", " three"]:
            yield {"token": word}
        yield {"done": True, "tokens": {"input": 3, "output": 3, "total": 6}}

    with patch("src.chat_core.main.stream_chat", side_effect=_fake_stream):
        response = client.post(
            "/chat/stream", json={"session_id": "t-004", "message": "count"}
        )

    events = [e for e in response.text.split("\n\n") if e.strip()]
    token_events = [json.loads(e[len("data: "):]) for e in events if '"token"' in e]
    assert len(token_events) == 3
    assert "".join(t["token"] for t in token_events) == "one two three"


# ── /tools ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_tools_unknown_tool_returns_400(client: TestClient) -> None:
    response = client.post(
        "/tools",
        json={"session_id": "t-005", "message": "Hi", "tools": ["nonexistent_tool"]},
    )
    assert response.status_code == 400


@pytest.mark.unit
def test_tools_known_tool_returns_200(client: TestClient) -> None:
    with patch("src.chat_core.pipeline.get_llm", return_value=_make_mock_llm()):
        response = client.post(
            "/tools",
            json={"session_id": "t-006", "message": "time?", "tools": ["get_current_time"]},
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "tool_calls" in data
    assert "tool_results" in data


@pytest.mark.unit
def test_tools_executes_tool_call(client: TestClient) -> None:
    """When the LLM returns a tool call, the server executes it and re-invokes."""
    first_msg = _mock_ai_message(
        "calling tool",
        tool_calls=[{"id": "tc-1", "name": "get_current_time", "args": {}}],
    )
    second_msg = _mock_ai_message("the time is now")

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[first_msg, second_msg])
    llm.bind_tools = MagicMock(return_value=llm)

    with patch("src.chat_core.pipeline.get_llm", return_value=llm):
        response = client.post(
            "/tools",
            json={"session_id": "t-007", "message": "time?", "tools": ["get_current_time"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["name"] == "get_current_time"
    assert len(data["tool_results"]) == 1
    assert data["answer"] == "the time is now"


# ── /reload-config ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_reload_config_returns_reloaded(client: TestClient) -> None:
    response = client.post("/reload-config")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reloaded"
    assert "provider" in data


# ── tools module unit tests ───────────────────────────────────────────────────


@pytest.mark.unit
def test_calculate_basic_arithmetic() -> None:
    from src.chat_core.tools import calculate

    assert calculate("2 + 3") == "5"
    assert calculate("10 - 4") == "6"
    assert calculate("3 * 7") == "21"
    assert calculate("10 / 4") == "2.5"


@pytest.mark.unit
def test_calculate_power_and_mod() -> None:
    from src.chat_core.tools import calculate

    assert calculate("2 ^ 10") == "1024"
    assert calculate("2 ** 10") == "1024"
    assert calculate("10 mod 3") == "1"
    assert calculate("10 % 3") == "1"


@pytest.mark.unit
def test_calculate_unary_minus() -> None:
    from src.chat_core.tools import calculate

    assert calculate("-5 * 2") == "-10"


@pytest.mark.unit
def test_calculate_division_by_zero() -> None:
    from src.chat_core.tools import calculate

    assert calculate("1 / 0").startswith("Error")


@pytest.mark.unit
def test_calculate_invalid_expression() -> None:
    from src.chat_core.tools import calculate

    assert calculate("import os").startswith("Error")
    assert calculate("open('file')").startswith("Error")
