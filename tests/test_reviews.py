import pytest

from app.reviews import load_reviews, parse_review, to_records

POST = """---
layout: post
title:  "Mr. Game and Sweat Reviews: Dota 2"
author: taylor
categories: [ review, game ]
---

Welcome back to Mr. Game and Sweat!

# Gameplay

**Game Review: Dota 2**

# Gameplay

Deep and strategic.

### Category Score: 3 out of 3

# Treadmill and Bike Compatibility

Hard on a treadmill.

### Category Score: 2 out of 3

# Computer Requirements

Runs on most things.

### Category Score: 2.5 out of 3

# Conclusion and Score

A fine game.

### Overall Score: (3+2+2.5)/3 = 2.5 our of 3

---

Mr. Game and Sweat Reviews: Dota 2 - By Taylor Dorsett
"""


def test_parses_game_slug_and_scores():
    review = parse_review(POST, "2024-07-07-review-dota-2.md")
    assert review.game == "Dota 2"
    assert review.url == "https://mrgameandsweat.com/review-dota-2/"
    assert (review.gameplay, review.treadmill, review.computer, review.overall) == (3, 2, 2.5, 2.5)


def test_sections_land_in_the_right_place_without_headings():
    sections = parse_review(POST, "2024-07-07-review-dota-2.md").sections
    assert "Deep and strategic." in sections["Gameplay"]
    assert "Hard on a treadmill." in sections["Treadmill and Bike Compatibility"]
    assert "Runs on most things." in sections["Computer Requirements"]
    assert sections["Conclusion"] == "A fine game."
    assert "#" not in "".join(sections.values())


def test_every_record_names_its_game():
    records = to_records(parse_review(POST, "2024-07-07-review-dota-2.md"))
    assert len(records) == 4
    assert all(r["chunk_text"].startswith("Dota 2 review, ") for r in records)
    assert len({r["id"] for r in records}) == 4
    assert "Treadmill and Bike Compatibility. Score: 2.0 out of 3." in records[1]["chunk_text"]


def test_post_missing_a_score_is_rejected():
    with pytest.raises(ValueError):
        parse_review(POST.replace("### Category Score: 2 out of 3\n", ""), "2024-07-07-x.md")


def test_real_reviews_parse_cleanly():
    reviews, skipped = load_reviews()
    assert not skipped
    if not reviews:
        pytest.skip("reviews aren't cloned")
    assert all(0 <= r.overall <= 3 for r in reviews)
    assert all(len(text) > 100 for r in reviews for text in r.sections.values())
