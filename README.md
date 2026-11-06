# Ask Game and Sweat

## What is this?

The chatbot behind the "Ask about a game" bubble on [Mr. Game and Sweat](https://mrgameandsweat.com), my site that reviews games you can play while exercising.

Ask it what to play on a treadmill and it searches the indexed reviews, answers from what it finds, and links you to the source.

**[How I built it](docs/README.md)** covers the model, how the reviews become searchable, the score filters, and how I check the answers.

## Why?

I am building LLM projects, and I wanted one on real data with real users. There are hundreds of reviews, which is way too much to paste into a prompt for every question, so it has to find the right ones first.

## How?

1. Every review gets split into its 4 sections and uploaded to Pinecone
2. The model turns your question into a search plus score filters
3. The sections it finds go to a free model on Groq, and the answer streams back to the bubble

The filters are the fun part. Text search finds relevant reviews, while score filters let a request for good treadmill games require a treadmill score of 3 out of 3.

This is retrieval augmented generation, or RAG. I built the search and application around an existing model. I did not train or fine tune the model on the reviews.

## Keeping costs down

The bubble is open to everyone and uses Groq's free plan. Provider limits can temporarily stop answers, so the app handles those errors and asks people to try again later.

These limits share the free tokens out fairly:

| Limit | Default |
| --- | --- |
| Messages per visitor | 10 per hour |
| Messages overall | 100 per day |
| Question length | 500 characters |
| Answer completion budget, including reasoning | 800 tokens |

The limits live in memory, so they reset if the server restarts. They help control traffic but are not a permanent spending cap. Check the provider account's current quotas and billing settings before changing plans.

## What I used to build this

| What | What it does |
| --- | --- |
| Python | The whole project |
| [Groq](https://groq.com) | Runs the model, `openai/gpt-oss-120b`, on the free plan |
| [Pinecone](https://www.pinecone.io) | Stores the review sections and searches them, using its `llama-text-embed-v2` embeddings |
| [OpenAI Python library](https://github.com/openai/openai-python) | Talks to Groq, which works with the same API |
| [FastAPI](https://fastapi.tiangolo.com) and [Uvicorn](https://www.uvicorn.org) | The API the bubble calls, including the streaming |
| [Render](https://render.com) | Free hosting for the API |
| [GitHub Actions](https://github.com/features/actions) | Runs the tests and uploads new reviews every day |
| [pytest](https://pytest.org) and [HTTPX](https://www.python-httpx.org) | The tests, plus my own evals to measure answer quality |
| PyYAML and python-dotenv | Reads the scores on each review and the keys in `.env` |
| [Mr. Game and Sweat](https://github.com/TaylorOD/mr-game-and-sweat) | The reviews, and the bubble lives on that site |
| [GitHub Copilot](https://github.com/features/copilot) | Reviews PRs |

## Running it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
git clone --depth 1 https://github.com/TaylorOD/mr-game-and-sweat data/site
```

Add your Groq and Pinecone keys to `.env`, then:

```
python -m scripts.upload_reviews
python chatbot.py
```

To run the API for the bubble:

```
uvicorn app.main:create_app --factory --port 8000
```

Run the tests with `pytest`. Run `python -m evals.run_evals` to measure answer quality, it uses up some of the free limit so it goes slowly.

## Hosting

`render.yaml` sets up the API on Render. It needs `GROQ_API_KEY` and `PINECONE_API_KEY`. It runs on the free plan, which sleeps after 15 minutes with no traffic and takes about a minute to wake up. Sleeping restarts the server, so it also resets the limits above.

A GitHub Action uploads new reviews every day. It needs `PINECONE_API_KEY` as a repo secret.
