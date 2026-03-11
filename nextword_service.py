import pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODELS_DIR = "models/next_word"

# Set by main.py during startup
model = None
tokenizer = None
max_seq_len = None

# Reverse index: {int_index: word}
_idx2word = None


def load_models():
    global model, tokenizer, max_seq_len, _idx2word

    model = tf.keras.models.load_model(f"{MODELS_DIR}/next_word_lstm_model.keras")

    with open(f"{MODELS_DIR}/next_word_tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(f"{MODELS_DIR}/max_seq_len.pkl", "rb") as f:
        max_seq_len = pickle.load(f)

    # Build reverse lookup once at startup (fast at inference time)
    _idx2word = {idx: word for word, idx in tokenizer.word_index.items()}

    print(f"[nextword_service] Model loaded. max_seq_len={max_seq_len}, vocab={len(_idx2word)}")


def _topk_sample(probs: np.ndarray, k: int = 10, temperature: float = 0.8) -> int:
    """Sample an index from the top-k probabilities using temperature scaling."""
    # Get indices of top-k probs
    top_k_indices = np.argsort(probs)[-k:]
    top_k_probs = probs[top_k_indices].astype("float64")

    # Apply temperature
    top_k_probs = np.log(top_k_probs + 1e-8) / temperature
    top_k_probs = np.exp(top_k_probs - np.max(top_k_probs))
    top_k_probs /= np.sum(top_k_probs)

    return int(np.random.choice(top_k_indices, p=top_k_probs))


def predict(text: str, top_n: int = 3, temperature: float = 0.8) -> dict:
    token_list = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")

    probs = model.predict(padded, verbose=0)[0]

    # ── Option A: Top-N words by raw probability ──────────────────────────
    top_indices = np.argsort(probs)[::-1][:top_n]
    top_words = [
        {
            "word": _idx2word.get(int(idx), "(unknown)"),
            "probability": round(float(probs[idx]) * 100, 2),
        }
        for idx in top_indices
        if int(idx) in _idx2word
    ]

    # ── Option B: Top-k temperature sampled word ──────────────────────────
    sampled_index = _topk_sample(probs, k=10, temperature=temperature)
    sampled_word = _idx2word.get(sampled_index, top_words[0]["word"] if top_words else "(unknown)")

    return {
        "next_word": sampled_word,           # B: creative sampled pick
        "top_words": top_words,              # A: top-3 deterministic picks
        "temperature": temperature,
    }
