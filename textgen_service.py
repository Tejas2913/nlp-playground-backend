import json
import pickle
import numpy as np
import tensorflow as tf

MODELS_DIR = "models/text_generator"

# Loaded at startup
_datasets = {}


def load_models():
    global _datasets

    for dataset in ["books", "shakespeare"]:
        if dataset == "books":
            model_path = f"{MODELS_DIR}/books/books_best_model.keras"
            char2idx_path = f"{MODELS_DIR}/books/books_char2idx.pkl"
            idx2char_path = f"{MODELS_DIR}/books/books_idx2char.pkl"
            config_path = f"{MODELS_DIR}/books/books_model_config.json"
        else:
            model_path = f"{MODELS_DIR}/shakespeare/best_text_model.keras"
            char2idx_path = f"{MODELS_DIR}/shakespeare/char2idx.pkl"
            idx2char_path = f"{MODELS_DIR}/shakespeare/idx2char.pkl"
            config_path = f"{MODELS_DIR}/shakespeare/model_config.json"

        model = tf.keras.models.load_model(model_path)

        with open(char2idx_path, "rb") as f:
            char2idx = pickle.load(f)
        with open(idx2char_path, "rb") as f:
            idx2char = pickle.load(f)
        with open(config_path, "r") as f:
            config = json.load(f)

        _datasets[dataset] = {
            "model": model,
            "char2idx": char2idx,
            "idx2char": idx2char,
            "seq_length": config["seq_length"],
            "vocab_size": config["vocab_size"],
        }

        print(f"[textgen_service] '{dataset}' model loaded. seq_length={config['seq_length']}")


def _sample_with_temperature(preds: np.ndarray, temperature: float) -> int:
    """Sample index from probability array using temperature."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds - np.max(preds))
    preds = exp_preds / np.sum(exp_preds)
    return int(np.random.choice(len(preds), p=preds))


def generate(seed_text: str, dataset: str, length: int, temperature: float) -> dict:
    if dataset not in _datasets:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'books' or 'shakespeare'.")

    ds = _datasets[dataset]
    model = ds["model"]
    char2idx = ds["char2idx"]
    idx2char = ds["idx2char"]
    seq_length = ds["seq_length"]

    # Prepare seed – use only known characters, lowercase
    seed = seed_text.lower()
    seed = "".join(c if c in char2idx else " " for c in seed)

    # Pad or trim to seq_length
    if len(seed) < seq_length:
        seed = seed.rjust(seq_length)
    else:
        seed = seed[-seq_length:]

    generated = seed_text  # Return seed + generated content
    current_seq = seed

    for _ in range(length):
        x = np.array([[char2idx.get(c, 0) for c in current_seq]])
        preds = model.predict(x, verbose=0)[0]
        next_idx = _sample_with_temperature(preds, temperature)
        next_char = idx2char.get(next_idx, "")
        generated += next_char
        current_seq = current_seq[1:] + next_char

    return {"generated_text": generated}
