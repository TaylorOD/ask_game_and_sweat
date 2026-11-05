from types import SimpleNamespace

from app import rag
from app.rag import SearchPlan, to_filter


def plan(**scores):
    empty = {"min_treadmill_score": None, "max_treadmill_score": None, "min_overall_score": None}
    return SearchPlan(query="q", **{**empty, **scores})


def test_no_scores_means_no_filter():
    assert to_filter(plan()) is None


def test_scores_become_pinecone_filters():
    assert to_filter(plan(min_treadmill_score=3, min_overall_score=2.7)) == {
        "treadmill_score": {"$gte": 3},
        "overall_score": {"$gte": 2.7},
    }
    assert to_filter(plan(max_treadmill_score=1.5)) == {"treadmill_score": {"$lte": 1.5}}


def test_retrieve_falls_back_when_the_filter_finds_nothing(monkeypatch):
    calls = []

    def fake_search(query, top_k, search_filter=None):
        calls.append(search_filter)
        return [] if search_filter else [{"game": "Brotato"}]

    monkeypatch.setattr(rag, "plan_search", lambda question: plan(min_treadmill_score=3))
    monkeypatch.setattr(rag, "search_reviews", fake_search)
    assert rag.retrieve("best treadmill game") == [{"game": "Brotato"}]
    assert calls == [{"treadmill_score": {"$gte": 3}}, None]


def test_planner_sees_earlier_turns_only_when_there_are_some():
    first = {"role": "user", "content": "best treadmill game?"}
    reply = {"role": "assistant", "content": "Brotato"}
    second = {"role": "user", "content": "what is its score?"}
    assert rag.search_question([first]) == "best treadmill game?"
    assert rag.search_question([first, reply, second]) == (
        "Earlier question: best treadmill game?\nEarlier answer: Brotato\nLatest question: what is its score?"
    )


def test_long_earlier_answers_are_trimmed_for_the_planner():
    long_reply = {"role": "assistant", "content": "x" * 5000}
    question = {"role": "user", "content": "q"}
    assert len(rag.search_question([question, long_reply, question])) < 600


def test_every_excerpt_carries_the_treadmill_score():
    chunk = {"url": "https://mrgameandsweat.com/hades/", "chunk_text": "Hades, Gameplay", "treadmill_score": 2.5}
    assert 'treadmill_and_bike_score="2.5"' in rag.system_prompt([chunk])


def test_plan_search_asks_for_a_strict_schema(monkeypatch):
    calls = []

    def create(**options):
        calls.append(options)
        content = '{"query": "chill", "min_treadmill_score": 3, "max_treadmill_score": null, "min_overall_score": null}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(rag, "model_client", lambda: fake)
    result = rag.plan_search("something chill")
    assert (result.query, result.min_treadmill_score) == ("chill", 3)
    schema = calls[0]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) == set(schema["schema"]["properties"])
