"""Frame sampling: decide how many frames to pull from a clip, and extract
them either at fixed intervals (Phase 1 MVP) or at scene-change points
(Phase 2 onward, falling back to uniform sampling when too few scene
changes are found).

Takes a `video_path: Path` directly rather than importing ingest.py, so
this module can be built and tested in isolation.
"""
from __future__ import annotations

import math
import subprocess
import uuid
from pathlib import Path

FFPROBE_TIMEOUT_SECONDS = 15
FFMPEG_TIMEOUT_SECONDS = 60


def frame_count_for_duration(duration_s: float, fmin: int = 8, fmax: int = 20) -> int:
    """Scale the number of frames to sample with clip length, clamped to
    [fmin, fmax].

    Formula: fmin + (fmax - fmin) * sqrt(duration_s / 60), clamped.
    Sqrt (rather than linear) scaling is used so short clips still get a
    reasonable frame budget and long clips don't dominate the runtime
    budget just because they're long — most of the marginal value in
    additional frames comes from the first ~minute of footage. Phase 2 may
    retune the curve shape/constants; the clamp and monotonicity contract
    (longer clips >= frames vs shorter ones) must be preserved.
    """
    if duration_s <= 0:
        return fmin
    if fmin > fmax:
        fmin, fmax = fmax, fmin

    scaled = fmin + (fmax - fmin) * math.sqrt(duration_s / 60.0)
    return max(fmin, min(fmax, round(scaled)))


def _run_ffmpeg(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def _sorted_frames(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("frame_*.jpg"))


def sample_uniform(video_path: Path, n: int, out_dir: Path) -> list[Path]:
    """Extract `n` frames at fixed, evenly-spaced timestamps across the
    clip. Used as the Phase 1 MVP extraction strategy, and as the fallback
    path for sample_keyframes.

    Returns a sorted list of frame image paths under a call-namespaced
    subdirectory of `out_dir` (so concurrent tasks never collide on
    filenames). Returns fewer than `n` paths only if the video is shorter
    than `n` distinct extractable frames.
    """
    if n <= 0:
        return []

    call_dir = out_dir / f"uniform_{uuid.uuid4().hex}"
    call_dir.mkdir(parents=True, exist_ok=True)

    try:
        probe = _run_ffmpeg(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            FFPROBE_TIMEOUT_SECONDS,
        )
        duration = float(probe.stdout.strip())
        if duration <= 0:
            return []
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return []

    # Evenly spaced timestamps, inset slightly from the very first/last
    # instant to avoid black frames or truncated GOPs at hard boundaries.
    inset = duration * 0.02
    span = max(duration - 2 * inset, 0.0)
    if n == 1:
        timestamps = [duration / 2]
    else:
        timestamps = [inset + span * i / (n - 1) for i in range(n)]

    frames: list[Path] = []
    for idx, ts in enumerate(timestamps):
        frame_path = call_dir / f"frame_{idx:03d}.jpg"
        try:
            proc = _run_ffmpeg(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{ts:.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(frame_path),
                ],
                FFMPEG_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if proc.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
            frames.append(frame_path)

    return _sorted_frames(call_dir)


def sample_keyframes(video_path: Path, n: int, out_dir: Path) -> list[Path]:
    """Extract up to `n` scene-change frames using ffmpeg's scene-detection
    filter. Falls back to `sample_uniform` if fewer than `n` scene-change
    frames are found (including on a static/no-scene-change clip).

    Returns a sorted list of frame paths under a call-namespaced
    subdirectory of `out_dir`.
    """
    if n <= 0:
        return []

    call_dir = out_dir / f"keyframes_{uuid.uuid4().hex}"
    call_dir.mkdir(parents=True, exist_ok=True)

    # Threshold picked empirically as a reasonable middle ground: high
    # enough to skip noise/compression artifacts, low enough to catch real
    # cuts in typical short-form clips.
    scene_threshold = 0.3
    pattern = call_dir / "frame_%03d.jpg"
    try:
        proc = _run_ffmpeg(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
                "-vsync", "vfr",
                "-start_number", "0",
                "-q:v", "2",
                "-frames:v", str(n),
                str(pattern),
            ],
            FFMPEG_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        proc = None

    frames = _sorted_frames(call_dir) if proc is not None else []

    if len(frames) >= n:
        return frames[:n]

    # Not enough (or any) scene changes found — fall back to uniform
    # sampling in a fresh subdirectory so we never return a mixed/partial
    # set of scene-change frames alongside uniform ones.
    return sample_uniform(video_path, n, out_dir)
