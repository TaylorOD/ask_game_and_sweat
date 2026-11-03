from app import rag
from app.rag import SearchPlan, to_filter
from evals.run_evals import score_in_text


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


def test_planner_sees_the_earlier_question_only_when_there_is_one():
    first = {"role": "user", "content": "best treadmill game?"}
    reply = {"role": "assistant", "content": "Brotato"}
    second = {"role": "user", "content": "what about Hades?"}
    assert rag.search_question([first]) == "best treadmill game?"
    assert rag.search_question([first, reply, second]) == (
        "Earlier question: best treadmill game?\nLatest question: what about Hades?"
    )


def test_score_check_ignores_the_scale():
    assert score_in_text("It scored 2 out of 3.", 2)
    assert not score_in_text("It scored 2 out of 3.", 3)
    assert not score_in_text("Treadmill: 2/3", 3)
    assert score_in_text("It scored 3 out of 3.", 3)
    assert score_in_text("It scored 3.0 out of 3.", 3)
