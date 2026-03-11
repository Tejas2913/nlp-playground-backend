from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# ─── Keras backward-compatibility patch ──────────────────────────────────────
# Models were saved with older Keras that stored 'batch_shape' in InputLayer
# config. Keras 2.12+ renamed this to 'batch_input_shape'. Patch from_config
# globally before any model is loaded so all three services benefit.
import tensorflow as tf
_orig_input_from_config = tf.keras.layers.InputLayer.from_config.__func__

@classmethod  # type: ignore[misc]
def _compat_input_from_config(cls, config):
    cfg = dict(config)
    if "batch_shape" in cfg:
        cfg["batch_input_shape"] = cfg.pop("batch_shape")
    return _orig_input_from_config(cls, cfg)

tf.keras.layers.InputLayer.from_config = _compat_input_from_config
# ─────────────────────────────────────────────────────────────────────────────

import sentiment_service
import nextword_service
import textgen_service



# ─── Startup / Shutdown ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup."""
    print("Loading sentiment models…")
    sentiment_service.load_models()

    print("Loading next-word model…")
    nextword_service.load_models()

    print("Loading text generation models…")
    textgen_service.load_models()

    print("✅ All models loaded. Server ready.")
    yield
    print("Shutting down…")


app = FastAPI(title="NLP Playground API", version="1.0.0", lifespan=lifespan)

# ─── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ──────────────────────────────────────────────

class SentimentRequest(BaseModel):
    text: str
    model: Literal["rnn", "lstm"] = "lstm"


class NextWordRequest(BaseModel):
    text: str
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)


class TextGenRequest(BaseModel):
    seed_text: str
    dataset: Literal["books", "shakespeare"] = "shakespeare"
    length: int = Field(default=200, ge=50, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "NLP Playground API is running"}


@app.post("/predict-sentiment")
def predict_sentiment(req: SentimentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        return sentiment_service.predict(req.text, req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-next-word")
def predict_next_word(req: NextWordRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        return nextword_service.predict(req.text, temperature=req.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-text")
def generate_text(req: TextGenRequest):
    if not req.seed_text.strip():
        raise HTTPException(status_code=400, detail="Seed text cannot be empty.")
    try:
        return textgen_service.generate(
            seed_text=req.seed_text,
            dataset=req.dataset,
            length=req.length,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
