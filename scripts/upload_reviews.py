"""Chunks every review and uploads it to Pinecone.

Run from the repo root: python -m scripts.upload_reviews
Run it again whenever new reviews come out, it overwrites records with the same id.
Use --days 7 to only upload recent reviews, which saves embedding quota.
"""
import argparse
import os
import time
from datetime import date, timedelta

from dotenv import load_dotenv
from pinecone import Pinecone

from app.reviews import load_reviews, to_records

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "game-and-sweat")
NAMESPACE = "reviews"
BATCH_SIZE = 90  # Pinecone doesn't allow more than 96 records at once


def upsert_with_retry(index, batch):
    for wait in (10, 30, 60, 60, None):
        try:
            return index.upsert_records(namespace=NAMESPACE, records=batch)
        except Exception as error:  # the free plan rate limits embedding
            if wait is None or "429" not in str(error):
                raise
            print(f"Rate limited, waiting {wait} seconds")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, help="only upload reviews from the last N days")
    args = parser.parse_args()

    reviews, skipped = load_reviews()
    if args.days is not None:
        cutoff = (date.today() - timedelta(days=args.days)).isoformat()
        reviews = [review for review in reviews if review.date >= cutoff]
    print(f"Parsed {len(reviews)} reviews, skipped {len(skipped)}")
    for reason in skipped:
        print(f"  skipped {reason}")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        print(f"Creating index {INDEX_NAME}")
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)

    index = pc.Index(INDEX_NAME)
    records = [record for review in reviews for record in to_records(review)]
    for start in range(0, len(records), BATCH_SIZE):
        upsert_with_retry(index, records[start : start + BATCH_SIZE])
        print(f"Uploaded {min(start + BATCH_SIZE, len(records))} of {len(records)}")


if __name__ == "__main__":
    main()
