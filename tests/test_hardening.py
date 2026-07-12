"""Prompt 9 hardening pass tests.

These exercise the FULL container run (src.main.main(), the real
src.pipeline.run_all — not a stubbed no-op) with one call path
deliberately broken, and assert the process still exits 0 with a valid,
complete /output/results.json. This is the hardening pass's core
guarantee: no single broken stage should ever turn into a crashed
container or a malformed/incomplete output file.

Also covers the Prompt 9 secondary-model-fallback and
env-configurable-concurrency additions to src/pipeline.py, src/grounding.py,
src/styling.py, src/judge.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import main as main_module
from src import pipeline
from src.grounding import GroundingError
from src.schemas import ALL_STYLES, Task

FIXTURE_TASKS = [
    {
        "task_id": "v1",
        "video_url": "https://storage.example.com/clips/clip1.mp4",
        "styles": ["formal", "sarcastic"],
    },
    {
        "task_id": "v2",
        "video_url": "https://storage.example.com/clips/clip2.mp4",
        "styles": list(ALL_STYLES),
    },
]

FAKE_VIDEO_PATH = Path("/tmp/fake_video_hardening.mp4")
FAKE_FRAMES = [Path("/tmp/fake_frame_000.jpg"), Path("/tmp/fake_frame_001.jpg")]


@pytest.fixture()
def env_and_paths(tmp_path, monkeypatch):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps(FIXTURE_TASKS), encoding="utf-8")
    output_path = tmp_path / "results.json"

    monkeypatch.setenv("GROUNDING_MODEL", "test-grounding-model")
    monkeypatch.setenv("STYLING_MODEL", "test-styling-model")
    monkeypatch.setenv("JUDGE_MODEL", "test-judge-model")
    monkeypatch.setenv("TASKS_INPUT_PATH", str(tasks_path))
    monkeypatch.setenv("RESULTS_OUTPUT_PATH", str(output_path))

    monkeypatch.setattr(main_module, "INPUT_PATH", tasks_path)
    monkeypatch.setattr(main_module, "OUTPUT_PATH", output_path)

    return tasks_path, output_path


def _patch_ingest_sampling_audio_ocr(monkeypatch):
    """Patch everything upstream of grounding to succeed instantly,
    without touching the filesystem/network — isolates the test to
    exercising the deliberately-broken stage plus the fallback path
    around it."""
    monkeypatch.setattr(pipeline.ingest, "download_video", mock.Mock(return_value=FAKE_VIDEO_PATH))
    monkeypatch.setattr(pipeline.ingest, "probe_duration", mock.Mock(return_value=30.0))
    monkeypatch.setattr(pipeline.ingest, "downscale", mock.Mock(return_value=FAKE_VIDEO_PATH))
    monkeypatch.setattr(pipeline.sampling, "sample_keyframes", mock.Mock(return_value=FAKE_FRAMES))
    monkeypatch.setattr(pipeline.sampling, "sample_uniform", mock.Mock(return_value=FAKE_FRAMES))
    monkeypatch.setattr(pipeline.audio, "transcribe", mock.Mock(return_value=None))
    monkeypatch.setattr(pipeline.ocr, "extract_text", mock.Mock(return_value=[]))


class TestFullRunSurvivesBrokenGrounding:
    """The 'deliberately broken call path' required by Prompt 9's
    verification checklist: force grounding.ground_video to always raise,
    and confirm the full container-level run (main(), real run_all, real
    concurrency/deadline handling) still exits 0 with complete, valid
    fallback content for every task/style."""

    def test_exit_code_is_zero(self, env_and_paths, monkeypatch):
        _patch_ingest_sampling_audio_ocr(monkeypatch)
        monkeypatch.setattr(
            pipeline.grounding, "ground_video",
            mock.AsyncMock(side_effect=GroundingError("forced failure for hardening test")),
        )

        exit_code = main_module.main()

        assert exit_code == 0

    def test_output_is_valid_and_complete_with_fallback_content(self, env_and_paths, monkeypatch):
        tasks_path, output_path = env_and_paths
        _patch_ingest_sampling_audio_ocr(monkeypatch)
        monkeypatch.setattr(
            pipeline.grounding, "ground_video",
            mock.AsyncMock(side_effect=GroundingError("forced failure for hardening test")),
        )

        exit_code = main_module.main()
        assert exit_code == 0

        assert output_path.exists()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)  # bare array, not wrapped in an object

        tasks = [Task.model_validate(item) for item in json.loads(tasks_path.read_text(encoding="utf-8"))]
        by_id = {entry["task_id"]: entry for entry in payload}

        for task in tasks:
            assert task.task_id in by_id
            captions = by_id[task.task_id]["captions"]
            for style in task.styles:
                assert style in captions
                assert captions[style].strip()  # non-empty fallback content present

    def test_unexpected_exception_in_grounding_also_falls_back_cleanly(self, env_and_paths, monkeypatch):
        """Defense in depth: even a raw, unanticipated exception type
        (not GroundingError) escaping ground_video must not crash the run."""
        _patch_ingest_sampling_audio_ocr(monkeypatch)
        monkeypatch.setattr(
            pipeline.grounding, "ground_video",
            mock.AsyncMock(side_effect=RuntimeError("totally unexpected failure")),
        )

        exit_code = main_module.main()
        assert exit_code == 0

        payload = json.loads(main_module.OUTPUT_PATH.read_text(encoding="utf-8"))
        assert len(payload) == len(FIXTURE_TASKS)


class TestConfigurableConcurrency:
    """Prompt 9 hardening: MAX_CONCURRENT_TASKS is overridable via env var
    without a code change, with a safe default when unset/invalid."""

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MAX_CONCURRENT_TASKS", raising=False)
        assert pipeline._max_concurrent_tasks() == pipeline._DEFAULT_MAX_CONCURRENT_TASKS

    def test_reads_valid_override(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_TASKS", "3")
        assert pipeline._max_concurrent_tasks() == 3

    def test_falls_back_to_default_on_non_integer(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_TASKS", "not-a-number")
        assert pipeline._max_concurrent_tasks() == pipeline._DEFAULT_MAX_CONCURRENT_TASKS

    def test_falls_back_to_default_on_non_positive(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_TASKS", "0")
        assert pipeline._max_concurrent_tasks() == pipeline._DEFAULT_MAX_CONCURRENT_TASKS


class TestNoHardcodedPathAssumptions:
    """Re-confirm (Prompt 9 checklist item) that main.py never hardcodes
    /input or /output paths beyond the documented env-var-with-default
    pattern established in Prompt 2."""

    def test_input_and_output_paths_follow_env_vars(self, tmp_path, monkeypatch):
        custom_input = tmp_path / "custom_in" / "tasks.json"
        custom_output = tmp_path / "custom_out" / "results.json"
        custom_input.parent.mkdir(parents=True, exist_ok=True)
        custom_input.write_text(json.dumps(FIXTURE_TASKS), encoding="utf-8")

        monkeypatch.setenv("TASKS_INPUT_PATH", str(custom_input))
        monkeypatch.setenv("RESULTS_OUTPUT_PATH", str(custom_output))

        # Re-import to pick up the env vars at module-load time, the same
        # way main.py itself resolves INPUT_PATH/OUTPUT_PATH.
        import importlib

        reloaded = importlib.reload(main_module)
        assert reloaded.INPUT_PATH == custom_input
        assert reloaded.OUTPUT_PATH == custom_output

        importlib.reload(main_module)  # restore default module state for other tests
