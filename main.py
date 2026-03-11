from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# ─── Keras backward-compatibility patch ──────────────────────────────────────
# Models were saved with Keras 3 which uses a newer serialization format.
# Keras 2.x (bundled with TF 2.15) doesn't know about:
#   1. 'batch_shape' in InputLayer config  → renamed to 'batch_input_shape'
#   2. 'DTypePolicy' objects for layer dtype → should be plain string e.g. 'float32'
import tensorflow as tf

# ── Fix 1: InputLayer batch_shape ────────────────────────────────────────────
_orig_input_from_config = tf.keras.layers.InputLayer.from_config.__func__

@classmethod  # type: ignore[misc]
def _compat_input_from_config(cls, config):
    cfg = dict(config)
    if "batch_shape" in cfg:
        cfg["batch_input_shape"] = cfg.pop("batch_shape")
    return _orig_input_from_config(cls, cfg)

tf.keras.layers.InputLayer.from_config = _compat_input_from_config

# ── Fix 2: DTypePolicy → register as a Keras-serializable class ──────────────
# Keras 3 serializes every layer's dtype as a DTypePolicy dict.
# We register a shim that Keras 2.x custom-object machinery can find.
@tf.keras.utils.register_keras_serializable(package="keras")
class DTypePolicy(tf.keras.mixed_precision.Policy):
    """Keras 3 compatibility shim — wraps tf.keras.mixed_precision.Policy."""
    @classmethod
    def from_config(cls, config):
        return cls(config.get("name", "float32"))

# ── Fix 3: Patch every layer's from_config to flatten DTypePolicy → str ──────
# Even with the registration above, some internal paths receive the raw dict
# before class lookup. This ensures any remaining 'dtype' dicts are resolved.
_orig_base_from_config = tf.keras.layers.Layer.from_config.__func__

def _flatten_dtype(cfg: dict) -> dict:
    """Recursively replace dtype DTypePolicy dicts with plain strings."""
    result = {}
    for k, v in cfg.items():
        if (k == "dtype" and isinstance(v, dict)
                and v.get("class_name") == "DTypePolicy"):
            result[k] = v.get("config", {}).get("name", "float32")
        elif isinstance(v, dict):
            result[k] = _flatten_dtype(v)
        elif isinstance(v, list):
            result[k] = [
                _flatten_dtype(i) if isinstance(i, dict) else i for i in v
            ]
        else:
            result[k] = v
    return result

@classmethod  # type: ignore[misc]
def _compat_layer_from_config(cls, config):
    return _orig_base_from_config(cls, _flatten_dtype(config))

tf.keras.layers.Layer.from_config = _compat_layer_from_config

# ── Fix 4: Patch legacy serializer that raises ValueError for DTypePolicy ─────
# The ACTUAL call site raising the error is class_and_config_for_serialized_keras_object
# in keras/src/saving/legacy/serialization.py. It encounters a Keras 3 format
# dict {'module':'keras','class_name':'DTypePolicy',...} and raises ValueError.
# We intercept it and return the matching Keras 2 Policy class directly.
try:
    from keras.src.saving.legacy import serialization as _legacy_ser
    _orig_class_and_config = _legacy_ser.class_and_config_for_serialized_keras_object

    def _patched_class_and_config(config, custom_objects=None,
                                   printable_module_name="object"):
        if (isinstance(config, dict)
                and config.get("class_name") == "DTypePolicy"):
            name = config.get("config", {}).get("name", "float32")
            return tf.keras.mixed_precision.Policy, {"name": name}
        return _orig_class_and_config(config, custom_objects, printable_module_name)

    _legacy_ser.class_and_config_for_serialized_keras_object = _patched_class_and_config
except Exception as _e:
    print(f"[compat] Could not patch legacy serializer: {_e}")
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
