# Video Captioning Agent

An advanced, multimodal video captioning pipeline designed to extract factual grounding descriptions from video frames and transcribe audio, style captions on-target across multiple custom rubric definitions, verify correctness using Chain-of-Verification (CoVe), and self-judge generated captions using a distinct evaluator model. 

Built specifically for high performance using **Fireworks AI** serverless endpoints.

---

## Architecture & Features

This system utilizes a modular, testable multi-stage pipeline:

1. **Ingestion & Sampling**: Robust downloading and chunk-based frame extraction using `ffmpeg` / `ffprobe` (capped to default visual range of 8-20 frames).
2. **Audio/OCR processing (Local Fallback)**: Runs Whisper for local audio transcription and OpenCV/RapidOCR for text extraction to feed additional context to the VLM.
3. **Stage 1: Grounding**: Generates a style-agnostic, factual description (`GroundingFacts`) using Fireworks' multimodal vision models via schema-enforced tool calling.
4. **Factual Verification**: A 3-step **Chain-of-Verification (CoVe)** pass that creates verification questions, answers them independently against the frames, and removes contradicted claims.
5. **Stage 2: Styling**: Adapts the factual description into highly styled captions concurrently based on defined rubrics (e.g. `formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`).
6. **Self-Judging & Quality Upgrade**: Evaluates styled captions using a distinct LLM on accuracy and style match, regenerating captions below quality thresholds (0.6).

---

## Fireworks AI Model Selection

To satisfy hackathon requirements, the pipeline runs exclusively via **Fireworks AI** using OpenAI-compatible SDK calls. The selected models are:

*   **Grounding Model**: `accounts/fireworks/models/llama4-maverick-instruct-basic`  
    *Natively multimodal 401B MoE with high visual intelligence and native function/tool calling support.*
*   **Styling Model**: `accounts/fireworks/models/deepseek-v3p1`  
    *State-of-the-art 674B MoE text model with exceptional instruction-following capability to align on-style captions.*
*   **Judge Model**: `accounts/fireworks/models/gpt-oss-120b`  
    *Distinct family open-weight model to eliminate self-preference bias when evaluating DeepSeek/LLaMA generated captions.*
*   **Fallbacks**: 
    *   Secondary Grounding: `accounts/fireworks/models/qwen2p5-vl-32b-instruct` (Vision fallback)
    *   Secondary Styling & Judging: `accounts/fireworks/models/gpt-oss-120b` / `accounts/fireworks/models/deepseek-v3p1`

---

## Setup Instructions

### 1. Prerequisites
Make sure you have the following installed on your system:
- Python 3.10+ (Recommended Python 3.14 / virtualenv)
- `ffmpeg` and `ffprobe` (for video downloading/processing)

### 2. Environment Setup
Clone the repository and initialize the python environment:
```bash
# Initialize using uv package manager
uv venv
source .venv/bin/activate

# Install the dependencies
uv pip install -r requirements.txt
```

*Note: If `uv` is not installed, standard pip can be used:*
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Credentials
Copy the example environment file:
```bash
cp .env.example .env
```
Open the `.env` file and input your actual Fireworks API key:
```bash
FIREWORKS_API_KEY=your_actual_fireworks_api_key_here
```

---

## Starting Steps

### Run the Pipeline
To run the full captioning pipeline for a specific video, invoke the orchestrator entrypoint (e.g. `pipeline.py` or configured script):

```bash
# Run captioning pipeline on a local video path or URL
uv run python -m src.pipeline --video-path path/to/your/video.mp4
```

### Running Tests
The project features a comprehensive unit testing suite mocking external API payloads. To run the tests:
```bash
uv run python -m pytest
```
or with standard pytest inside your activated virtualenv:
```bash
pytest
```
