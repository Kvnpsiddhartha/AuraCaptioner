"""Shared test fixtures for the Phase 1 modules (ingest/sampling/audio/ocr).

All fixtures are generated locally via ffmpeg/espeak-ng (already required by
the Docker image / dev environment) rather than depending on any external
network resource, so these tests run offline and deterministically.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def multi_cut_video(fixtures_dir: Path) -> Path:
    """~8s clip with 3 hard scene cuts (red/blue/green/yellow) and a sine
    tone audio track. Enough cuts to exercise sample_keyframes' real
    scene-detection path (not just its uniform-sampling fallback)."""
    out = fixtures_dir / "multi_cut.mp4"
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2,format=yuv420p",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2,format=yuv420p",
        "-f", "lavfi", "-i", "color=c=green:s=320x240:d=2,format=yuv420p",
        "-f", "lavfi", "-i", "color=c=yellow:s=320x240:d=2,format=yuv420p",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
        "-map", "[v]", "-map", "4:a", "-shortest",
        "-c:v", "libx264", "-c:a", "aac", str(out),
        "-loglevel", "error",
    ])
    return out


@pytest.fixture(scope="session")
def static_silent_video(fixtures_dir: Path) -> Path:
    """5s single-color clip with no audio track at all: no scene changes,
    no speech."""
    out = fixtures_dir / "static_silent.mp4"
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=5,format=yuv420p",
        "-an", "-c:v", "libx264", str(out),
        "-loglevel", "error",
    ])
    return out


@pytest.fixture(scope="session")
def speech_video(fixtures_dir: Path) -> Path:
    """A clip with real synthesized speech (espeak-ng), for has_speech's
    positive case. Skips (rather than fails) if espeak-ng isn't installed,
    since it's a test-only convenience, not a runtime dependency."""
    if shutil.which("espeak-ng") is None:
        pytest.skip("espeak-ng not available to synthesize a speech fixture")

    wav_path = fixtures_dir / "speech.wav"
    _run(["espeak-ng", "This is a short test clip with clear speech audio.", "-w", str(wav_path)])

    out = fixtures_dir / "speech_clip.mp4"
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=white:s=320x240:d=6,format=yuv420p",
        "-i", str(wav_path),
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out),
        "-loglevel", "error",
    ])
    return out


@pytest.fixture(scope="session")
def text_frame(fixtures_dir: Path) -> Path:
    """A single frame with clear, high-contrast on-screen text."""
    out = fixtures_dir / "text_frame.jpg"
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=white:s=640x480",
        "-vf", "drawtext=text='HELLO WORLD CAPTION':fontsize=48:fontcolor=black:x=50:y=200",
        "-frames:v", "1", str(out),
        "-loglevel", "error",
    ])
    return out


@pytest.fixture(scope="session")
def blank_frame(fixtures_dir: Path) -> Path:
    """A single frame with no text: flat color, nothing to OCR."""
    out = fixtures_dir / "blank_frame.jpg"
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=gray:s=640x480",
        "-frames:v", "1", str(out),
        "-loglevel", "error",
    ])
    return out
