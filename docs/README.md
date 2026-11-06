# How I built Ask Game and Sweat

I wanted someone to land on Mr. Game and Sweat, ask what to play during a workout, and get an answer they could actually use. The site has hundreds of reviews. Finding the right game should not mean opening twenty tabs.

[Try the bot](https://mrgameandsweat.com/) by clicking **Ask about a game**.

This is a public RAG application built around the site's reviews. RAG means retrieval augmented generation. It finds relevant source material first, then gives that material to a language model to write the answer. In this case, the sources are game reviews with actual treadmill scores and links you can check.

## What model is it?

The default is `openai/gpt-oss-120b`, served through Groq. It is an existing reasoning language model from OpenAI with publicly available weights. Groq runs it, so this project does not need its own GPU. [Groq's model documentation](https://console.groq.com/docs/model/openai/gpt-oss-120b) covers the underlying model.

The app uses it twice. First it turns a question into a structured search plan. Then it writes an answer from the review sections that search returns. The OpenAI Python library talks to Groq's compatible API. These chatbot requests go to Groq.

A separate model, Pinecone's `llama-text-embed-v2`, handles embeddings. Those are numerical representations of text that let the search find related ideas even when the wording is different.

## How did I train it?

I did not train the base model or fine tune its weights. My work is the retrieval system around it: preparing the reviews, making them searchable, choosing what context to send, setting answer rules, and checking the results.

The reviews are a knowledge base. They are not a training dataset for this project. When a new review gets added, I update the search index. The model itself stays the same. Chat messages are used for the current conversation and are not fed into a training job by this app.

That fits this site well. The content keeps growing, and an answer needs to reflect the published review and its score. Updating the source material is more useful here than trying to bake those facts into model weights.

## From a review to an answer

1. **Split the review by meaning.** Each review becomes four sections: gameplay, treadmill and bike compatibility, computer requirements, and conclusion. Each section carries the game name, scores, and review URL.
2. **Make the sections searchable.** Pinecone embeds the text and stores it alongside the score metadata. A question can find relevant sections without sending the whole site to the model.
3. **Plan the search.** The model returns a validated JSON search plan. It separates the topic from score requirements, including minimum treadmill and overall scores.
4. **Retrieve the evidence.** The default search returns five sections. If the filters find nothing, the app retries without them. Five sections can include more than one section from the same game.
5. **Write and stream the answer.** The model gets the excerpts, treadmill scores, source URLs, and instructions to answer only from that material. FastAPI streams the text to the website as it arrives.

Short follow ups get context from the previous question and answer, so something like "what was its score?" can still produce a useful search.

## The part that made recommendations better

Semantic search finds text that sounds relevant. A high treadmill score is a separate requirement.

For "what works well while running?", the planner can ask Pinecone for reviews with a treadmill score of 3 out of 3. For "top rated", it can also require an overall score of at least 2.7. A question about one specific game does not automatically get those filters.

That lets the recommendation use both the meaning of the question and the site's scoring system. The model is not responsible for guessing a score from a game's reputation.

## How I check it

The [eval script](../evals/run_evals.py) checks three things separately: whether retrieval finds the right game, whether an answer states the right treadmill score, and what happens to recommendation scores when filters are added.

It uses a seeded sample of 30 reviews for retrieval and 10 for answer checks. It compares answers with retrieval against answers without it. A model judge extracts the stated score, which is then compared with the parsed review score.

Those are focused checks, not proof that every recommendation is right. The judge can make mistakes too. I also check the actual chat in a browser, including source links, mobile layout, loading states, and errors. The automated tests cover the parser, search planning, API, and usage limits.

## Keeping it current and usable

A daily GitHub Action checks reviews added in the last seven days and uploads their sections. Stable record IDs let those uploads replace matching records. Older edits need a full upload, and this job does not automatically remove deleted reviews from the index.

The API runs on Render's free plan. Opening the chat starts a wake request, and the bubble explains the delay if the server is asleep. The app limits each visitor to 10 messages per hour and the service to 100 per day. Questions are capped at 500 characters. The answer request has an 800 token completion budget, which also includes model reasoning.

Those request limits live in memory and reset when the service restarts. They help control traffic, but they are not a permanent spending cap. Provider quotas and account settings still matter.

The model is instructed to stay within the reviews and say when they do not cover a question. That reduces unsupported answers, but it cannot guarantee accuracy. The review links are there so people can check the source.

## Why I built it

I am building LLM projects that people can actually use. This one connects a real content library to a public website, with retrieval, structured search, streaming, evaluations, and deployment all working together.

The thing I like most is that the difficult part is visible in a simple question. Someone asks what to play on a treadmill, and the app has to find the right evidence, use the right score, and give them somewhere useful to go next.

[Back to setup and running instructions](../README.md)
