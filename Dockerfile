FROM python:3.11-slim

# ffmpeg is required from Prompt 3A onward (ingest.py, sampling.py, audio.py)
# for downloading/probing/downscaling video and frame/audio extraction.
# libglib2.0-0/libgomp1/libsm6/libxext6/libxrender1 are the small set of
# shared libs opencv-python-headless and onnxruntime (rapidocr) need on a
# slim base even though neither pulls in a GUI/X11 stack. Installed now so
# later prompts (3D onward) don't need Dockerfile changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        gcc \
        libc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Mount points. The harness is expected to mount /input/tasks.json and
# /output/ — these env vars exist so main.py never hardcodes a path that
# would break if the harness mounts elsewhere; the defaults below match
# the documented contract.
ENV TASKS_INPUT_PATH=/input/tasks.json
ENV RESULTS_OUTPUT_PATH=/output/results.json

CMD ["python", "-m", "src.main"]
