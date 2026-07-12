"""Audio pipeline: a cheap voice-activity gate followed by transcription.

`has_speech` must be cheap (energy/VAD-based, no model load) so silent
clips never pay for a transcription call. `transcribe` re-uses that gate
internally and only invokes the (comparatively expensive) speech-to-text
model when there's something worth transcribing.

Both functions return safe defaults (False / None) rather than raising —
silence and "no transcript" are always acceptable answers for this stage.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Optional

VAD_SAMPLE_RATE_HZ = 16_000
VAD_FRAME_MS = 30  # webrtcvad only accepts 10/20/30ms frames
VAD_FRAME_BYTES = int(VAD_SAMPLE_RATE_HZ * (VAD_FRAME_MS / 1000.0)) * 2  # 16-bit mono
# Fraction of VAD-positive frames required to call a clip "has speech".
# Low bar on purpose: we only need to know whether transcription is worth
# attempting at all, not how much speech there is.
VAD_SPEECH_FRAME_RATIO_THRESHOLD = 0.10
# 0 (least aggressive) .. 3 (most aggressive) filtering of non-speech.
VAD_AGGRESSIVENESS = 2

FFMPEG_TIMEOUT_SECONDS = 60

# Whisper model size is a local implementation detail of this module (not
# a Settings field — Settings is frozen per the project schema contract),
# overridable via env var for hosts with different memory/latency budgets.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")

_whisper_model = None  # lazily constructed, cached across calls in-process


def _extract_pcm_wav(video_path: Path) -> Optional[Path]:
    """Extract mono 16kHz 16-bit PCM audio to a temp .wav file, or None on
    failure (including: no audio stream present)."""
    out_path = video_path.with_name(f"{video_path.stem}_{uuid.uuid4().hex}.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(VAD_SAMPLE_RATE_HZ),
        "-f", "wav",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        return None
    return out_path


def _iter_vad_frames(pcm_bytes: bytes) -> list[bytes]:
    frames = []
    for start in range(0, len(pcm_bytes) - VAD_FRAME_BYTES + 1, VAD_FRAME_BYTES):
        frames.append(pcm_bytes[start:start + VAD_FRAME_BYTES])
    return frames


def has_speech(video_path: Path) -> bool:
    """Lightweight voice-activity check: extract audio, run WebRTC VAD over
    30ms frames, and return True if enough frames are flagged as speech.

    Gates the expensive transcription step — this must stay cheap. Returns
    False (never raises) on any failure: missing audio track, ffmpeg
    failure, corrupt file, or a missing webrtcvad dependency all count as
    "no speech detected", which is the safe default.
    """
    wav_path = None
    try:
        wav_path = _extract_pcm_wav(video_path)
        if wav_path is None:
            return False

        import webrtcvad  # optional dependency; absence == "no speech"

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != VAD_SAMPLE_RATE_HZ:
                return False
            pcm_bytes = wf.readframes(wf.getnframes())

        frames = _iter_vad_frames(pcm_bytes)
        if not frames:
            return False

        speech_frames = sum(
            1 for frame in frames if vad.is_speech(frame, VAD_SAMPLE_RATE_HZ)
        )
        ratio = speech_frames / len(frames)
        return ratio >= VAD_SPEECH_FRAME_RATIO_THRESHOLD
    except Exception:
        return False
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


def _load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # optional heavy dependency

        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe(video_path: Path) -> Optional[str]:
    """Return a transcript string for `video_path`, or None if there's no
    speech, transcription isn't available, or the call fails.

    Calls has_speech() first and returns None immediately (without loading
    any model or making a network call) if it's False.
    """
    if not has_speech(video_path):
        return None

    wav_path = None
    try:
        wav_path = _extract_pcm_wav(video_path)
        if wav_path is None:
            return None

        model = _load_whisper_model()
        segments, _info = model.transcribe(str(wav_path))
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text or None
    except Exception:
        return None
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
