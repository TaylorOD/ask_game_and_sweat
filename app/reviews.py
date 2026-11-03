import re
from dataclasses import dataclass
from pathlib import Path

import yaml

POSTS_DIR = Path(__file__).parent.parent / "data" / "site" / "_posts"
SITE_URL = "https://mrgameandsweat.com"

# Headings vary a lot between reviews, but every review has these score lines in the same order
CATEGORY_SCORE = re.compile(r"(?mi)^#+\s*Category Score:\s*([\d.]+)\s*out of\s*3.*$")
OVERALL_SCORE = re.compile(r"(?mi)^#+\s*Overall Score:.*?([\d.]+)\s*(?:out|our) of\s*3.*$")
SECTION_NAMES = ["Gameplay", "Treadmill and Bike Compatibility", "Computer Requirements"]
HEADING = re.compile(r"(?m)^#+ .*$")


@dataclass
class Review:
    game: str
    slug: str
    date: str
    gameplay: float
    treadmill: float
    computer: float
    overall: float
    sections: dict

    @property
    def url(self):
        return f"{SITE_URL}/{self.slug}/"


def clean(text):
    text = HEADING.sub("", text).replace("**", "")
    text = re.sub(r"(?m)^---\s*$", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_review(text, filename):
    """Returns a Review, or raises ValueError if the post doesn't have the usual shape."""
    _, front, body = text.split("---\n", 2)
    meta = yaml.safe_load(front)
    game = meta["title"].split("Reviews:", 1)[-1].strip()
    date, slug = filename[:10], filename[11:].removesuffix(".md")

    scores = list(CATEGORY_SCORE.finditer(body))
    overall = OVERALL_SCORE.search(body)
    if len(scores) != 3 or not overall:
        raise ValueError(f"{filename}: {len(scores)} category scores, overall {bool(overall)}")

    sections = {}
    start = 0
    for name, score in zip(SECTION_NAMES, scores):
        sections[name] = clean(body[start : score.start()])
        start = score.end()
    sections["Conclusion"] = clean(body[start : overall.start()])

    return Review(
        game=game,
        slug=slug,
        date=date,
        gameplay=float(scores[0].group(1)),
        treadmill=float(scores[1].group(1)),
        computer=float(scores[2].group(1)),
        overall=float(overall.group(1)),
        sections=sections,
    )


def load_reviews(posts_dir=POSTS_DIR):
    reviews, skipped = [], []
    for path in sorted(Path(posts_dir).glob("*.md")):
        try:
            reviews.append(parse_review(path.read_text(encoding="utf-8"), path.name))
        except (ValueError, KeyError) as error:
            skipped.append(str(error))
    return reviews, skipped


def to_records(review):
    """One Pinecone record per section. The game name goes in the text so every chunk stands alone."""
    section_scores = {
        "Gameplay": review.gameplay,
        "Treadmill and Bike Compatibility": review.treadmill,
        "Computer Requirements": review.computer,
        "Conclusion": review.overall,
    }
    records = []
    for i, (section, text) in enumerate(review.sections.items()):
        records.append({
            "id": f"{review.slug}-{i}",
            "chunk_text": (
                f"{review.game} review, {section}. Score: {section_scores[section]} out of 3. "
                f"Overall score: {review.overall} out of 3.\n\n{text}"
            ),
            "game": review.game,
            "slug": review.slug,
            "url": review.url,
            "section": section,
            "gameplay_score": review.gameplay,
            "treadmill_score": review.treadmill,
            "computer_score": review.computer,
            "overall_score": review.overall,
        })
    return records
