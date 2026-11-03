import json
import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app import rag
from app.limits import RateLimiter

log = logging.getLogger("game_and_sweat")

MAX_QUESTION_CHARS = 500
ONE_HOUR = 60 * 60
ONE_DAY = 24 * ONE_HOUR
DEFAULT_ORIGINS = "https://mrgameandsweat.com,https://www.mrgameandsweat.com"
DAILY_LIMIT_MESSAGE = "I've hit my limit for today. Come back tomorrow, or browse the reviews in the meantime!"
VISITOR_LIMIT_MESSAGE = "That's a lot of questions! Give me a little while and try again."


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=3000)


class ChatRequest(BaseModel):
    messages: list[Turn] = Field(min_length=1, max_length=10)


def client_ip(request):
    # On a host, the proxy adds the real address last. Anything before that can be faked by the visitor.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and os.getenv("TRUST_PROXY", "false").lower() == "true":
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def create_app():
    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]
    per_visitor = RateLimiter(int(os.getenv("MESSAGES_PER_HOUR", "10")), ONE_HOUR)
    overall = RateLimiter(int(os.getenv("MESSAGES_PER_DAY", "100")), ONE_DAY)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request):
        # Scripts can fake this, it just stops other websites from using the bot. The limits below do the real work.
        if request.headers.get("origin") not in origins:
            raise HTTPException(403, "This chat only works on Mr. Game and Sweat.")

        messages = [turn.model_dump() for turn in body.messages]
        question = messages[-1]
        if question["role"] != "user" or len(question["content"]) > MAX_QUESTION_CHARS:
            raise HTTPException(422, f"Questions can be up to {MAX_QUESTION_CHARS} characters.")

        # Check the visitor first, so one person going over doesn't use up the daily cap
        if not per_visitor.allow(client_ip(request)):
            raise HTTPException(429, VISITOR_LIMIT_MESSAGE)
        if not overall.allow("all"):
            raise HTTPException(429, DAILY_LIMIT_MESSAGE)

        def events():
            try:
                for event in rag.stream_answer(messages):
                    yield json.dumps(event) + "\n"
            except Exception:
                log.exception("Chat failed")
                message = "Something went wrong on my end. Please try again."
                yield json.dumps({"type": "error", "message": message}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/healthz")
    def health():
        return Response("ok", media_type="text/plain")

    return app
