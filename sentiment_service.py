import pickle
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 200
MODELS_DIR = "models/sentiment"

# These will be set by main.py during startup
rnn_model = None
lstm_model = None
tokenizer = None


def load_models():
    global rnn_model, lstm_model, tokenizer

    rnn_model = tf.keras.models.load_model(f"{MODELS_DIR}/rnn_model.h5", compile=False)
    lstm_model = tf.keras.models.load_model(f"{MODELS_DIR}/lstm_model.h5", compile=False)

    with open(f"{MODELS_DIR}/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    print("[sentiment_service] Models loaded.")


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text


def predict(text: str, model_name: str) -> dict:
    cleaned = preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH)

    model = lstm_model if model_name.lower() == "lstm" else rnn_model
    score = float(model.predict(padded, verbose=0)[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"
    return {"sentiment": sentiment, "score": round(score, 4)}
