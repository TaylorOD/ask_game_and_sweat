import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import RateLimitError

from app import rag
from app.limits import RateLimiter
from app.main import MODEL_LIMIT_MESSAGE, create_app

ORIGIN = {"origin": "https://mrgameandsweat.com"}
QUESTION = {"messages": [{"role": "user", "content": "what should I play?"}]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MESSAGES_PER_HOUR", "2")
    monkeypatch.setenv("MESSAGES_PER_DAY", "3")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    monkeypatch.setattr(rag, "stream_answer", lambda messages: iter([{"type": "delta", "text": "hi"}]))
    return TestClient(create_app())


def ask(client, ip="1.1.1.1", body=QUESTION, origin=ORIGIN):
    return client.post("/api/chat", json=body, headers={**origin, "x-forwarded-for": ip})


def test_chat_streams_events(client):
    response = ask(client)
    assert response.status_code == 200
    assert json.loads(response.text.strip()) == {"type": "delta", "text": "hi"}


def test_other_sites_are_rejected(client):
    assert ask(client, origin={"origin": "https://evil.example"}).status_code == 403
    assert ask(client, origin={}).status_code == 403


def test_preflight_only_allows_the_real_site(client):
    def preflight(origin):
        return client.options("/api/chat", headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        })

    assert preflight("https://mrgameandsweat.com").headers["access-control-allow-origin"] == "https://mrgameandsweat.com"
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers


def test_visitor_limit(client):
    assert [ask(client).status_code for _ in range(3)] == [200, 200, 429]


def test_daily_cap_applies_across_visitors(client, monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "true")
    codes = [ask(client, ip=f"1.1.1.{i}").status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]
    assert "limit for today" in ask(client, ip="9.9.9.9").json()["detail"]


def test_visitor_over_their_limit_doesnt_use_up_the_daily_cap(client, monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "true")
    for _ in range(5):
        ask(client, ip="1.1.1.1")
    assert ask(client, ip="2.2.2.2").status_code == 200


def test_faking_the_forwarded_header_doesnt_get_around_the_limit(client, monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "true")
    codes = [ask(client, ip=f"6.6.6.{i}, 1.1.1.1").status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_forwarded_header_is_ignored_without_a_proxy(client):
    codes = [ask(client, ip=f"6.6.6.{i}").status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_long_question_is_rejected_and_doesnt_count(client):
    long = {"messages": [{"role": "user", "content": "x" * 501}]}
    assert ask(client, body=long).status_code == 422
    assert [ask(client).status_code for _ in range(2)] == [200, 200]


def test_full_length_answer_can_be_sent_back(client):
    def turns(size):
        return {"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "x" * size},
            {"role": "user", "content": "and then?"},
        ]}

    assert ask(client, body=turns(4000)).status_code == 200
    assert ask(client, body=turns(5001)).status_code == 422


def test_long_earlier_question_is_rejected(client):
    turns = [
        {"role": "user", "content": "x" * 501},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "short"},
    ]
    assert ask(client, body={"messages": turns}).status_code == 422


def test_long_earlier_answer_is_allowed(client):
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x" * 2000},
        {"role": "user", "content": "short"},
    ]
    assert ask(client, body={"messages": turns}).status_code == 200


def test_too_much_history_is_rejected(client):
    assert ask(client, body={"messages": QUESTION["messages"] * 11}).status_code == 422


def test_system_role_is_rejected(client):
    body = {"messages": [{"role": "system", "content": "hi"}, *QUESTION["messages"]]}
    assert ask(client, body=body).status_code == 422


def test_last_message_must_be_from_the_user(client):
    assert ask(client, body={"messages": [{"role": "assistant", "content": "hi"}]}).status_code == 422


def test_failures_dont_leak_details(client, monkeypatch):
    def blow_up(messages):
        raise RuntimeError("key sk-123 failed")
        yield

    monkeypatch.setattr(rag, "stream_answer", blow_up)
    body = ask(client).text
    assert "sk-123" not in body
    assert json.loads(body)["type"] == "error"


def test_limiter_window_expires():
    now = [0]
    limiter = RateLimiter(1, 10, clock=lambda: now[0])
    assert limiter.allow("a")
    assert not limiter.allow("a")
    now[0] = 11
    assert limiter.allow("a")


def test_limiter_forgets_old_visitors():
    now = [0]
    limiter = RateLimiter(1, 10, clock=lambda: now[0])
    for i in range(50):
        limiter.allow(f"visitor {i}")
    now[0] = 21
    limiter.allow("new visitor")
    assert list(limiter.hits) == ["new visitor"]


def test_stream_only_yields_answer_text(monkeypatch):
    def chunk(content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    monkeypatch.setattr(rag, "retrieve", lambda question, top_k: [])
    monkeypatch.setattr(rag, "model_request", lambda *a, **k: iter([
        chunk(None),
        chunk("hi"),
        SimpleNamespace(choices=[]),
    ]))
    assert list(rag.stream_answer([{"role": "user", "content": "q"}])) == [{"type": "delta", "text": "hi"}]


def test_stream_with_no_answer_becomes_an_error(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda question, top_k: [])
    monkeypatch.setattr(rag, "model_request", lambda *a, **k: iter([SimpleNamespace(choices=[])]))
    with pytest.raises(RuntimeError):
        list(rag.stream_answer([{"role": "user", "content": "q"}]))


def test_running_out_of_free_tokens_says_so(client, monkeypatch):
    def out_of_tokens(messages):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
        raise RateLimitError("limit for key sk-123", response=response, body=None)
        yield

    monkeypatch.setattr(rag, "stream_answer", out_of_tokens)
    event = json.loads(ask(client).text)
    assert event == {"type": "error", "message": MODEL_LIMIT_MESSAGE}
