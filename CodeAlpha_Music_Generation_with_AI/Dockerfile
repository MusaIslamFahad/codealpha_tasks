# ── AI Music Generation Studio — Dockerfile ──────────────────────────────────
# Build : docker build -t music-gen-studio .
# Run   : docker run -p 7860:7860 --gpus all music-gen-studio
#
# For CPU-only: remove --gpus all
# For HuggingFace Spaces: this Dockerfile is used automatically.

FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────────────────────
COPY . .

# ── Output directory ──────────────────────────────────────────────────────────
RUN mkdir -p outputs

# ── Non-root user (security best practice) ───────────────────────────────────
RUN useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Expose Gradio port ────────────────────────────────────────────────────────
EXPOSE 7860

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
