import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="6sense.ai Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Example(BaseModel):
    input: str
    output: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or instruction")
    creativity: float = Field(0.7, ge=0.0, le=1.0, description="Controls randomness (temperature)")
    reasoning: bool = Field(True, description="If true, include a short reasoning and examples when helpful")
    examples: Optional[List[Example]] = Field(
        default=None,
        description="Optional few‑shot examples to steer the model"
    )


class GenerateResponse(BaseModel):
    output: str
    model: str
    provider: str = "pollinations.ai"


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # noqa: BLE001
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:  # noqa: BLE001
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


POLLINATIONS_TEXT_URL = "https://text.pollinations.ai/"
POLLINATIONS_DEFAULT_MODEL = os.getenv("POLLINATIONS_MODEL", "llama-3.1-8b-instruct")


def build_messages(req: GenerateRequest) -> list:
    system_parts = [
        "You are 6sense.ai, a creative, helpful AI agent.",
        "Respond with clear structure: a quick answer, brief reasoning, and 1–2 concrete examples when helpful.",
        "Prefer concise paragraphs and bullet points over long walls of text.",
    ]
    if not req.reasoning:
        system_parts.append("If reasoning is not requested, keep it short and focus on the final answer.")

    system_prompt = "\n".join(system_parts)

    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    if req.examples:
        for ex in req.examples:
            messages.append({"role": "user", "content": ex.input})
            messages.append({"role": "assistant", "content": ex.output})

    messages.append({"role": "user", "content": req.prompt})
    return messages


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate creative text using Pollinations.ai open models (free, no key)."""
    messages = build_messages(req)

    payload = {
        "model": POLLINATIONS_DEFAULT_MODEL,
        "messages": messages,
        "temperature": req.creativity,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # Pollinations supports both GET and POST. We use POST for structured messages.
            r = await client.post(POLLINATIONS_TEXT_URL, json=payload)
            r.raise_for_status()
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}

            # The API often returns { text: "..." } or { response: "..." }
            output = data.get("text") or data.get("response") or r.text

            if not output:
                raise HTTPException(status_code=502, detail="Empty response from model")

            return GenerateResponse(output=output.strip(), model=POLLINATIONS_DEFAULT_MODEL)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Provider error: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
