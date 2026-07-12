"""Tests for src/ingest.py — download, probe, downscale."""
from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
import requests

from src.ingest import IngestError, download_video, downscale, probe_duration


def _probe_dims(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    return stream["width"], stream["height"]


@pytest.fixture()
def local_file_server(tmp_path):
    """Serve tmp_path over HTTP on localhost so download_video can be
    exercised against a real (if local) HTTP response, without depending
    on any external network resource."""
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=str(tmp_path), **kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", tmp_path
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestDownloadVideo:
    def test_downloads_file_successfully(self, local_file_server, tmp_path):
        base_url, serve_dir = local_file_server
        (serve_dir / "clip.mp4").write_bytes(b"fake-mp4-bytes" * 100)

        dest_dir = tmp_path / "downloads"
        result = download_video(f"{base_url}/clip.mp4", dest_dir)

        assert result.exists()
        assert result.suffix == ".mp4"
        assert result.read_bytes() == b"fake-mp4-bytes" * 100

    def test_raises_ingest_error_not_raw_exception_on_bad_host(self, tmp_path):
        with pytest.raises(IngestError):
            download_video("http://127.0.0.1:1/does-not-exist.mp4", tmp_path / "downloads")

    def test_raises_ingest_error_on_404(self, local_file_server, tmp_path):
        base_url, _serve_dir = local_file_server
        with pytest.raises(IngestError):
            download_video(f"{base_url}/missing.mp4", tmp_path / "downloads")

    def test_error_message_identifies_the_url(self, tmp_path):
        bad_url = "http://127.0.0.1:1/bad.mp4"
        with pytest.raises(IngestError, match="127.0.0.1:1/bad.mp4"):
            download_video(bad_url, tmp_path / "downloads")

    def test_never_leaks_raw_requests_exception(self, tmp_path):
        # requests.RequestException (and its subclasses, e.g. ConnectionError)
        # must always be normalized to IngestError, never propagate raw.
        try:
            download_video("http://127.0.0.1:1/bad.mp4", tmp_path / "downloads")
            pytest.fail("expected IngestError")
        except IngestError:
            pass
        except requests.RequestException:
            pytest.fail("raw requests exception leaked past download_video")


class TestProbeDuration:
    def test_returns_positive_duration_for_known_clip(self, multi_cut_video):
        duration = probe_duration(multi_cut_video)
        assert duration > 0
        assert duration == pytest.approx(8.0, abs=0.5)

    def test_raises_ingest_error_for_missing_file(self, tmp_path):
        with pytest.raises(IngestError):
            probe_duration(tmp_path / "does-not-exist.mp4")

    def test_raises_ingest_error_for_unparseable_input(self, tmp_path):
        garbage = tmp_path / "garbage.mp4"
        garbage.write_bytes(b"not a real video file")
        with pytest.raises(IngestError):
            probe_duration(garbage)


class TestDownscale:
    def test_output_is_smaller_resolution(self, multi_cut_video, tmp_path):
        work_copy = tmp_path / multi_cut_video.name
        shutil.copy(multi_cut_video, work_copy)

        original_w, original_h = _probe_dims(work_copy)
        out_path = downscale(work_copy, max_height=120)

        assert out_path.exists()
        out_w, out_h = _probe_dims(out_path)
        assert out_h <= 120
        assert out_h < original_h

    def test_raises_ingest_error_on_missing_input(self, tmp_path):
        with pytest.raises(IngestError):
            downscale(tmp_path / "does-not-exist.mp4", max_height=240)

    def test_no_upscaling_when_already_small(self, multi_cut_video, tmp_path):
        work_copy = tmp_path / multi_cut_video.name
        shutil.copy(multi_cut_video, work_copy)
        original_w, original_h = _probe_dims(work_copy)

        out_path = downscale(work_copy, max_height=original_h + 1000)
        out_w, out_h = _probe_dims(out_path)

        assert out_h == original_h
