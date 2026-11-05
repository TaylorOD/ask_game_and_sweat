# Ask Game and Sweat

## What is this?

The chatbot behind the "Ask about a game" bubble on [Mr. Game and Sweat](https://mrgameandsweat.com), my site that reviews games you can play while exercising.

It has read every review on the site. Ask it what to play on a treadmill and it answers from the reviews and links you to them.

## Why?

I am building LLMs and LLM projects, and I wanted one on real data with real users. There are almost 400 reviews, which is way too much to paste into a prompt for every question, so it has to find the right ones first.

## How?

1. Every review gets split into its 4 sections and uploaded to Pinecone
2. The model turns your question into a search plus score filters
3. The sections it finds go to a free model on Groq, and the answer streams back to the bubble

The filters are the fun part. Text search can find a treadmill game but it can't rank by score. Asking for the best treadmill games used to find games averaging 2.4 out of 3. With filters they average 3.0.

## What tech?

| Tech | What it does |
| --- | --- |
| Python | The whole project |
| FastAPI and Uvicorn | The API behind the bubble |
| OpenAI `gpt-4.1-mini` | Plans the search and writes the answer |
| Pinecone | Stores the review sections and searches them |
| `llama-text-embed-v2` | The embedding model, run by Pinecone |
| Render | Hosts the API |
| GitHub Actions | Runs the tests and uploads new reviews every day |
| pytest | Tests |

The reviews come from [Mr. Game and Sweat](https://github.com/TaylorOD/mr-game-and-sweat). Evals are my own script in `evals/`.

## Keeping costs down

The bubble is open to everyone, so it runs on Groq's free plan with no card on file. There is no bill to run up. If the free limit runs out, the bubble says so and people try again later.

These limits share the free tokens out fairly:

| Limit | Default |
| --- | --- |
| Messages per visitor | 10 per hour |
| Messages overall | 100 per day |
| Question length | 500 characters |
| Answer length | 800 tokens |

The limits live in memory, so they reset if the server restarts. The real ceiling is Groq's free plan, which was about 8000 tokens a minute and 200000 a day when I built this.

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
