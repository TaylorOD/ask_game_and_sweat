"""Measures how well the bot does. Uses up some of the free Groq limit, so it goes slowly.

Run from the repo root: python -m evals.run_evals
"""
import random
import statistics
import time

from pydantic import BaseModel, ConfigDict

from app.rag import DEFAULT_TOP_K, answer, parse_json, search_reviews
from app.reviews import load_reviews

SEED = 7
LOOKUP_COUNT = 30
RETRIEVAL_TOP_KS = [1, 3, 5, 10, 20]
ANSWER_COUNT = 10
ANSWER_TOP_KS = [DEFAULT_TOP_K]
PAUSE_SECONDS = 40  # the free plan allows about 8000 tokens a minute

RECOMMENDATIONS = [
    "What's the best game to play on a treadmill?",
    "I want something easy to play while I'm running. What should I get?",
    "What game works great on an exercise bike?",
    "Recommend a chill game for a long walk on the treadmill.",
    "What are your top rated games for working out?",
    "Which games are too hard to play while exercising?",
]


def lookup_question(review):
    return f"What treadmill and bike score did {review.game} get?"


class StatedScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treadmill_score: float | None


def stated_treadmill_score(answer_text):
    """An LLM judge, since numbers in an answer can also be game titles or other scores."""
    return parse_json(
        StatedScore,
        """Read this answer about a video game. Return the treadmill and
        bike compatibility score it states, as a number out of 3. Ignore game titles
        and any other scores, like gameplay or overall. Return null if it doesn't
        state one.""",
        answer_text,
    ).treadmill_score


def retrieval_eval(sample):
    print(f"\nRetrieval: is the right game in the results? ({len(sample)} questions)")
    for top_k in RETRIEVAL_TOP_KS:
        hits = sum(
            any(c["game"] == review.game for c in search_reviews(lookup_question(review), top_k))
            for review in sample
        )
        print(f"  top {top_k:>2}: {hits / len(sample):.0%}")


def answer_eval(sample):
    print(f"\nAnswers: did it state the right treadmill score? ({len(sample)} questions)")
    for top_k in [None, *ANSWER_TOP_KS]:
        right = 0
        for review in sample:
            messages = [{"role": "user", "content": lookup_question(review)}]
            text, _ = answer(messages, top_k=top_k or 1, use_rag=top_k is not None)
            right += stated_treadmill_score(text) == review.treadmill
            time.sleep(PAUSE_SECONDS)
        label = "no RAG" if top_k is None else f"top {top_k:>2}"
        print(f"  {label}: {right / len(sample):.0%}")


def recommendation_eval(reviews):
    best = sum(r.treadmill == 3 for r in reviews)
    average = statistics.mean(r.treadmill for r in reviews)
    print(f"\nRecommendations: {best} of {len(reviews)} reviews score 3 on treadmill, average {average:.2f}")
    for question in RECOMMENDATIONS:
        plain = {c["game"]: c["treadmill_score"] for c in search_reviews(question, 10)}
        text, chunks = answer([{"role": "user", "content": question}])
        planned = {c["game"]: c["treadmill_score"] for c in chunks}
        print(f"\n  {question}")
        print(f"    average treadmill score found, plain search: {statistics.mean(plain.values()):.2f}")
        print(f"    average treadmill score found, with filters: {statistics.mean(planned.values()):.2f}, games: {len(planned)}")
        print("    " + text.replace("\n", "\n    "))
        time.sleep(PAUSE_SECONDS)


def main():
    reviews, _ = load_reviews()
    sample = random.Random(SEED).sample(reviews, LOOKUP_COUNT)
    retrieval_eval(sample)
    answer_eval(sample[:ANSWER_COUNT])
    recommendation_eval(reviews)


if __name__ == "__main__":
    main()
