# ── Stage 1: dependency install ──────────────────────────────────────────────
# Use a separate builder stage so compile-time tools (gcc, libc-dev) are
# never baked into the final runtime image.
FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
# Install into an isolated prefix so we can COPY just the site-packages into
# the final stage without dragging along pip/wheel/setuptools.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM python:3.11-slim

# Runtime-only native libs:
#   ffmpeg          – video download/probe/downscale/frame-extract (ingest.py,
#                     sampling.py, audio.py)
#   libglib2.0-0    – GLib runtime, pulled by ffmpeg's glib-based filters
#   libgomp1        – OpenMP runtime, required by some ffmpeg codec threads
# NOTE: libsm6 / libxext6 / libxrender1 (OpenCV X11 stubs) and gcc/libc-dev
#       (build tools for webrtcvad) are no longer needed because those
#       packages have been removed from requirements.txt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Carry over only the installed Python packages from the builder stage.
COPY --from=builder /install /usr/local

# Faster Python startup: skip .pyc generation and buffer stdout/stderr so
# log lines appear immediately in the harness log stream.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/ ./src/

# ── Pipeline tuning ───────────────────────────────────────────────────────────
# Reserve ~60 s for image-pull + container startup + output write, leaving
# 540 s of pure compute budget under the 600 s hard cap.
ENV TASKS_INPUT_PATH=/input/tasks.json \
    RESULTS_OUTPUT_PATH=/output/results.json \
    MAX_RUNTIME_SECONDS=540 \
    # Reduce frame count: 16 frames gives excellent VLM coverage while
    # cutting per-task image-upload payload vs the previous 20-frame default.
    FRAMES_MAX=16 \
    # Chain-of-Verification adds 3 sequential LLM round-trips per task.
    # Disabled by default to stay comfortably inside the runtime budget.
    # Set ENABLE_VERIFICATION=true to re-enable for higher-accuracy runs.
    ENABLE_VERIFICATION=false \
    # Self-judge adds up to (1 + max_retries) LLM calls per style.
    # Keep enabled but cap retries at 1 (was 2) to halve judge overhead.
    MAX_SELF_JUDGE_RETRIES=1

CMD ["python", "-m", "src.main"]
