"""Video ingest: download, probe, and downscale source clips.

Every public function in this module is a hard boundary: it must never let
a raw exception (requests, subprocess, OSError, ...) escape uncaught.
Failures are normalized into `IngestError` so pipeline.py can catch one
well-known exception type for this stage and fall back gracefully.

This module does not import ingest-adjacent modules (sampling/audio/ocr) —
it only produces a local video Path for them to consume.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

# Connect/read timeout for the download (seconds). Applied to both legs of
# the request so a stalled connection can't hang the task indefinitely.
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
# Cap on downloaded bytes as a defensive measure against a misbehaving or
# malicious video_url filling the disk. 500MB is generous for a short clip.
DOWNLOAD_MAX_BYTES = 500 * 1024 * 1024

FFPROBE_TIMEOUT_SECONDS = 15
FFMPEG_TIMEOUT_SECONDS = 120


class IngestError(Exception):
    """Raised for any failure in downloading, probing, or downscaling a video."""


def _suffix_from_url(url: str) -> str:
    """Best-effort file extension from the URL path, defaulting to .mp4."""
    path = urlparse(url).path
    suffix = Path(path).suffix
    if suffix and len(suffix) <= 5 and suffix[1:].isalnum():
        return suffix
    return ".mp4"


def download_video(url: str, dest_dir: Path) -> Path:
    """Download `url` into `dest_dir`, returning the local path.

    Raises IngestError (never a raw requests/urllib exception) on any
    network failure, timeout, non-2xx status, oversized, or empty response.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{uuid.uuid4().hex}{_suffix_from_url(url)}"

    bytes_written = 0
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(DOWNLOAD_TIMEOUT_SECONDS, DOWNLOAD_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    bytes_written += len(chunk)
                    if bytes_written > DOWNLOAD_MAX_BYTES:
                        raise IngestError(
                            f"video_url={url!r} exceeded max download size "
                            f"({DOWNLOAD_MAX_BYTES} bytes)"
                        )
                    f.write(chunk)
    except IngestError:
        dest_path.unlink(missing_ok=True)
        raise
    except requests.RequestException as exc:
        dest_path.unlink(missing_ok=True)
        raise IngestError(f"failed to download video_url={url!r}: {exc}") from exc
    except OSError as exc:
        dest_path.unlink(missing_ok=True)
        raise IngestError(
            f"failed to write downloaded file for video_url={url!r}: {exc}"
        ) from exc

    if bytes_written == 0:
        dest_path.unlink(missing_ok=True)
        raise IngestError(f"downloaded empty file for video_url={url!r}")

    return dest_path


def probe_duration(path: Path) -> float:
    """Return the duration of the video at `path` in seconds via ffprobe.

    Raises IngestError if ffprobe fails to run, exits non-zero, or its
    output can't be parsed into a positive float.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise IngestError(f"ffprobe failed to run on {path}: {exc}") from exc

    if proc.returncode != 0:
        raise IngestError(
            f"ffprobe exited {proc.returncode} for {path}: {proc.stderr.strip()[-500:]}"
        )

    output = proc.stdout.strip()
    try:
        duration = float(output)
    except ValueError as exc:
        raise IngestError(
            f"ffprobe returned unparseable duration {output!r} for {path}"
        ) from exc

    if duration <= 0:
        raise IngestError(f"ffprobe returned non-positive duration {duration} for {path}")

    return duration


def downscale(path: Path, max_height: int = 480) -> Path:
    """Downscale `path` so its height is at most `max_height`, preserving
    aspect ratio. Purpose: keep frame extraction and any upload payloads
    small and fast for downstream stages.

    Returns the path to the (re-encoded) copy, namespaced next to the
    source file. Raises IngestError on failure. If the source is already
    at or below `max_height`, ffmpeg leaves the frame size unchanged (the
    filter is a conditional no-op) — we still re-encode so callers get a
    single, predictable codec/container downstream.
    """
    out_path = path.with_name(f"{path.stem}_ds{path.suffix}")

    # Only scale down, never up; force even width via -2 (required by most
    # encoders) while leaving height untouched when already <= max_height.
    scale_filter = (
        f"scale='if(gt(ih,{max_height}),-2,iw)':'if(gt(ih,{max_height}),{max_height},ih)'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise IngestError(f"ffmpeg downscale failed to run on {path}: {exc}") from exc

    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise IngestError(
            f"ffmpeg downscale failed for {path}: {proc.stderr.strip()[-500:]}"
        )

    return out_path
