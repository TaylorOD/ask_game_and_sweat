import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.budget import BudgetUnavailable, MemoryBudget, PostgresBudget, create_budget
from app import main, rag


def test_reserves_all_limits_or_none():
    budget = MemoryBudget()
    assert budget.reserve([("daily", "all", 1, 86400)]) is None
    assert budget.reserve([("visitor", "a", 1, 3600), ("daily", "all", 1, 86400)]) == "daily"
    assert budget.reserve([("visitor", "a", 1, 3600)]) is None


def test_missing_required_database_fails_startup(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_DATABASE_URL", raising=False)
    monkeypatch.setenv("REQUIRE_PERSISTENT_LIMITS", "true")
    with pytest.raises(RuntimeError, match="RATE_LIMIT_DATABASE_URL"):
        create_budget("test")


def test_database_outage_prevents_model_call(monkeypatch):
    class Down:
        def reserve(self, rules):
            raise BudgetUnavailable()
    monkeypatch.setattr(main, "create_budget", lambda _: Down())
    def unexpected(*args, **kwargs):
        pytest.fail("Model must not be called during a quota-store outage")
    monkeypatch.setattr(rag, "stream_answer", unexpected)
    client = TestClient(main.create_app())
    response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]},
                           headers={"origin": "https://mrgameandsweat.com"})
    assert response.status_code == 503
    assert "password" not in response.text


@pytest.fixture
def postgres_budget():
    url = os.getenv("TEST_RATE_LIMIT_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_RATE_LIMIT_DATABASE_URL for real PostgreSQL integration tests")
    namespace = "test-" + uuid.uuid4().hex
    with psycopg.connect(url) as conn:
        conn.execute((Path(__file__).parents[1] / "sql/chat_budget.sql").read_text())
    yield url, namespace
    with psycopg.connect(url) as conn:
        conn.execute("DELETE FROM chat_budget_hits WHERE namespace=%s", (namespace,))


def test_postgres_restart_concurrency_and_expiry(postgres_budget):
    url, namespace = postgres_budget
    rules = [("daily", "all", 7, 86400)]
    def reserve(_):
        return PostgresBudget(url, namespace).reserve(rules)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(reserve, range(24)))
    assert results.count(None) == 7
    assert results.count("daily") == 17
    # A fresh object/connection models a new process after restart.
    assert PostgresBudget(url, namespace).reserve(rules) == "daily"
    with psycopg.connect(url) as conn:
        conn.execute("UPDATE chat_budget_hits SET expires_at=clock_timestamp()-interval '1 second' "
                     "WHERE namespace=%s", (namespace,))
    assert PostgresBudget(url, namespace).reserve(rules) is None


def test_postgres_atomic_buckets_and_no_raw_subjects(postgres_budget):
    url, namespace = postgres_budget
    budget = PostgresBudget(url, namespace)
    assert budget.reserve([("daily", "all", 1, 86400)]) is None
    assert budget.reserve([("visitor", "192.0.2.1", 1, 3600), ("daily", "all", 1, 86400)]) == "daily"
    assert budget.reserve([("visitor", "192.0.2.1", 1, 3600)]) is None
    with psycopg.connect(url) as conn:
        subjects = conn.execute("SELECT subject FROM chat_budget_hits WHERE namespace=%s", (namespace,)).fetchall()
    assert all(len(subject) == 64 and subject != "192.0.2.1" for (subject,) in subjects)


def test_postgres_errors_are_sanitized(monkeypatch):
    def down(*args, **kwargs):
        raise psycopg.OperationalError("password=super-secret")
    monkeypatch.setattr(psycopg, "connect", down)
    with pytest.raises(BudgetUnavailable) as error:
        PostgresBudget("not-a-real-credential", "test").reserve([])
    assert "super-secret" not in str(error.value)
