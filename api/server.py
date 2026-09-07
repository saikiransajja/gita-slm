"""
FastAPI server for the Gita SLM.

Endpoints:
  GET  /            → health check + model info
  POST /meaning     → generate meaning for a verse
  GET  /examples    → list sample verses to try

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K, API_HOST, API_PORT
from src.inference.generate import get_engine


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class MeaningRequest(BaseModel):
    verse: str = Field(
        ...,
        min_length=5,
        description="Sanskrit shloka or transliterated verse text",
        example="karmanye vadhikaraste ma phaleshu kadachana",
    )
    language: str = Field(
        default="both",
        description="Output language: 'en', 'hi', or 'both'",
    )
    max_new_tokens: int = Field(
        default=MAX_NEW_TOKENS,
        ge=10,
        le=512,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=TEMPERATURE,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more creative)",
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=200,
        description="Top-k sampling filter",
    )


class MeaningResponse(BaseModel):
    verse:      str
    meaning_en: str
    meaning_hi: str
    language:   str


class HealthResponse(BaseModel):
    status:     str
    model:      str
    parameters: int
    device:     str


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Gita SLM API",
    description=(
        "A Small Language Model trained on the Bhagavad Gita. "
        "Provide a Sanskrit verse (shloka) and receive its meaning "
        "in English and Hindi."
    ),
    version="1.0.0",
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns model status and metadata."""
    engine = get_engine()
    return HealthResponse(
        status="ok",
        model="Gita SLM — GPT-style transformer",
        parameters=engine.model.count_parameters(),
        device=engine.device,
    )


@app.post("/meaning", response_model=MeaningResponse, tags=["Inference"])
def get_meaning(request: MeaningRequest):
    """
    Generate the meaning of a Bhagavad Gita verse (shloka).

    **Input:** Sanskrit or transliterated verse text
    **Output:** English and Hindi meanings
    """
    if request.language not in ("en", "hi", "both"):
        raise HTTPException(
            status_code=422,
            detail="language must be one of: 'en', 'hi', 'both'",
        )

    engine = get_engine()

    result = engine.generate_meaning(
        verse=request.verse,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        language=request.language,
    )

    # Filter output based on requested language
    meaning_en = result["meaning_en"] if request.language in ("en", "both") else ""
    meaning_hi = result["meaning_hi"] if request.language in ("hi", "both") else ""

    return MeaningResponse(
        verse=result["verse"],
        meaning_en=meaning_en,
        meaning_hi=meaning_hi,
        language=request.language,
    )


@app.get("/examples", tags=["Examples"])
def get_examples():
    """Returns a list of example verses you can try."""
    return {
        "examples": [
            {
                "chapter": 2,
                "verse": 47,
                "shloka": "karmanye vadhikaraste ma phaleshu kadachana",
                "description": "The most famous verse — on duty without attachment to results",
            },
            {
                "chapter": 2,
                "verse": 20,
                "shloka": "na jayate mriyate va kadachin nayam bhutva bhavita va na bhuyah",
                "description": "The soul is eternal — it neither births nor dies",
            },
            {
                "chapter": 4,
                "verse": 7,
                "shloka": "yada yada hi dharmasya glanir bhavati bharata",
                "description": "Whenever dharma declines, I appear",
            },
            {
                "chapter": 18,
                "verse": 66,
                "shloka": "sarva-dharman parityajya mam ekam sharanam vraja",
                "description": "Surrender unto Me alone",
            },
            {
                "chapter": 6,
                "verse": 5,
                "shloka": "uddhared atmanatmanam natmanam avasadayet",
                "description": "Elevate yourself by your own mind",
            },
        ]
    }


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host=API_HOST, port=API_PORT, reload=False)
