# ── Base image: Python 3.11 slim ─────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs as a non-root user; create app dir with correct permissions
RUN useradd -m -u 1000 user
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source + model files ─────────────────────────────────────────────────
COPY --chown=user . .

# Switch to non-root user (required by HF Spaces)
USER user

# ── HF Spaces expects port 7860 ───────────────────────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
