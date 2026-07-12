"""Tests for src/pipeline.py — full per-task orchestration and run_all's
deadline/concurrency behavior."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline
from src.config import Settings
from src.grounding import GroundingError
from src.ingest import IngestError
from src.schemas import ALL_STYLES, GroundingFacts, Task, VerificationResult

FAKE_VIDEO_PATH = Path("/tmp/fake_video.mp4")
FAKE_FRAMES = [Path("/tmp/frame_000.jpg"), Path("/tmp/frame_001.jpg")]

FACTS = GroundingFacts(
    subjects=["a dog"], actions=["running"], setting="a park",
    mood="playful", on_screen_text=[], audible_speech=None,
    notable_sounds=["barking"],
)


def _settings(**overrides) -> Settings:
    base = dict(
        grounding_model="test-grounding-model",
        styling_model="test-styling-model",
        judge_model="test-judge-model",
        fallback_caption="A short video clip.",
        max_self_judge_retries=2,
        frames_min=8,
        frames_max=20,
        max_runtime_seconds=570,
    )
    base.update(overrides)
    return Settings(**base)


def _task(task_id="t1", styles=None) -> Task:
    return Task(
        task_id=task_id,
        video_url="https://storage.example.com/clip.mp4",
        styles=styles or ["formal", "sarcastic"],
    )


def _patch_happy_path(monkeypatch, *, extra_captions: dict | None = None):
    """Patch every pipeline stage to succeed quickly, without touching the
    filesystem or network."""
    monkeypatch.setattr(pipeline.ingest, "download_video", mock.Mock(return_value=FAKE_VIDEO_PATH))
    monkeypatch.setattr(pipeline.ingest, "probe_duration", mock.Mock(return_value=30.0))
    monkeypatch.setattr(pipeline.ingest, "downscale", mock.Mock(return_value=FAKE_VIDEO_PATH))
    monkeypatch.setattr(pipeline.sampling, "sample_keyframes", mock.Mock(return_value=FAKE_FRAMES))
    monkeypatch.setattr(pipeline.sampling, "sample_uniform", mock.Mock(return_value=FAKE_FRAMES))
    monkeypatch.setattr(pipeline.audio, "transcribe", mock.Mock(return_value=None))
    monkeypatch.setattr(pipeline.ocr, "extract_text", mock.Mock(return_value=[]))
    monkeypatch.setattr(pipeline.grounding, "ground_video", mock.AsyncMock(return_value=FACTS))
    monkeypatch.setattr(
        pipeline.verification, "verify_facts",
        mock.AsyncMock(return_value=VerificationResult(
            original_facts=FACTS, verification_questions=[], verification_answers=[],
            cleaned_facts=FACTS, dropped_claims=[],
        )),
    )

    async def _style_all(facts, styles, settings):
        captions = {style: f"caption for {style}" for style in styles}
        if extra_captions:
            captions.update(extra_captions)
        return captions

    monkeypatch.setattr(pipeline.styling, "style_all", _style_all)

    async def _judge_and_regenerate(facts, captions, settings, regenerate_fn, max_retries):
        return captions  # no-op judge by default: pass captions through unchanged

    monkeypatch.setattr(pipeline.judge, "judge_and_regenerate", _judge_and_regenerate)


class TestRunTaskHappyPath:
    def test_end_to_end_returns_all_requested_styles(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        task = _task(styles=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"])

        result = asyncio.run(pipeline.run_task(task, _settings()))

        assert result.task_id == task.task_id
        assert set(result.captions.keys()) == set(task.styles)
        for text in result.captions.values():
            assert text.strip()

    def test_judge_and_regenerate_output_is_used(self, monkeypatch):
        _patch_happy_path(monkeypatch)

        async def _judge_upgrades(facts, captions, settings, regenerate_fn, max_retries):
            return {style: f"UPGRADED {text}" for style, text in captions.items()}

        monkeypatch.setattr(pipeline.judge, "judge_and_regenerate", _judge_upgrades)

        task = _task(styles=["formal"])
        result = asyncio.run(pipeline.run_task(task, _settings()))

        assert result.captions["formal"].startswith("UPGRADED")


class TestRunTaskFallbacks:
    def test_ingest_failure_falls_back_for_this_task_only(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            pipeline.ingest, "download_video",
            mock.Mock(side_effect=IngestError("simulated download failure")),
        )
        task = _task(styles=["formal", "sarcastic"])
        settings = _settings()

        result = asyncio.run(pipeline.run_task(task, settings))

        assert result.task_id == task.task_id
        assert set(result.captions.keys()) == set(task.styles)
        for text in result.captions.values():
            assert text == settings.fallback_caption

    def test_grounding_failure_falls_back(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            pipeline.grounding, "ground_video",
            mock.AsyncMock(side_effect=GroundingError("simulated grounding failure")),
        )
        task = _task(styles=["formal", "sarcastic"])
        settings = _settings()

        result = asyncio.run(pipeline.run_task(task, settings))

        for text in result.captions.values():
            assert text == settings.fallback_caption

    def test_no_frames_extracted_falls_back(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(pipeline.sampling, "sample_keyframes", mock.Mock(return_value=[]))
        monkeypatch.setattr(pipeline.sampling, "sample_uniform", mock.Mock(return_value=[]))
        task = _task(styles=["formal"])
        settings = _settings()

        result = asyncio.run(pipeline.run_task(task, settings))

        assert result.captions["formal"] == settings.fallback_caption

    def test_verification_failure_still_produces_captions(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            pipeline.verification, "verify_facts",
            mock.AsyncMock(side_effect=RuntimeError("simulated verification crash")),
        )
        task = _task(styles=["formal"])

        result = asyncio.run(pipeline.run_task(task, _settings()))

        assert result.captions["formal"] == "caption for formal"

    def test_styling_failure_falls_back(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            pipeline.styling, "style_all",
            mock.AsyncMock(side_effect=RuntimeError("simulated styling crash")),
        )
        task = _task(styles=["formal", "sarcastic"])
        settings = _settings()

        result = asyncio.run(pipeline.run_task(task, settings))

        for text in result.captions.values():
            assert text == settings.fallback_caption

    def test_judge_failure_keeps_pre_judge_captions(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        monkeypatch.setattr(
            pipeline.judge, "judge_and_regenerate",
            mock.AsyncMock(side_effect=RuntimeError("simulated judge crash")),
        )
        task = _task(styles=["formal"])

        result = asyncio.run(pipeline.run_task(task, _settings()))

        assert result.captions["formal"] == "caption for formal"

    def test_completely_unexpected_exception_still_returns_fallback(self, monkeypatch):
        monkeypatch.setattr(
            pipeline.ingest, "download_video",
            mock.Mock(side_effect=RuntimeError("totally unexpected boom")),
        )
        task = _task(styles=["formal", "sarcastic"])
        settings = _settings()

        result = asyncio.run(pipeline.run_task(task, settings))

        assert result.task_id == task.task_id
        for text in result.captions.values():
            assert text == settings.fallback_caption


class TestJudgeDeadlineSkip:
    def test_judge_skipped_when_close_to_deadline(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        judge_spy = mock.AsyncMock(return_value={"formal": "SHOULD NOT BE CALLED"})
        monkeypatch.setattr(pipeline.judge, "judge_and_regenerate", judge_spy)

        task = _task(styles=["formal"])
        near_deadline = time.monotonic() + (pipeline.JUDGE_MIN_SECONDS_REQUIRED - 5.0)

        result = asyncio.run(pipeline.run_task(task, _settings(), deadline=near_deadline))

        judge_spy.assert_not_called()
        assert result.captions["formal"] == "caption for formal"

    def test_judge_runs_when_plenty_of_time_remains(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        judge_spy = mock.AsyncMock(return_value={"formal": "judged text"})
        monkeypatch.setattr(pipeline.judge, "judge_and_regenerate", judge_spy)

        task = _task(styles=["formal"])
        far_deadline = time.monotonic() + 300.0

        result = asyncio.run(pipeline.run_task(task, _settings(), deadline=far_deadline))

        judge_spy.assert_called_once()
        assert result.captions["formal"] == "judged text"


class TestRunAll:
    def test_all_tasks_succeed_concurrently(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        tasks = [_task(task_id=f"t{i}") for i in range(3)]
        deadline = time.monotonic() + 60.0

        results = asyncio.run(pipeline.run_all(tasks, _settings(), deadline))

        assert {r.task_id for r in results} == {t.task_id for t in tasks}

    def test_one_task_failing_does_not_affect_others(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        call_count = {"n": 0}
        real_download = pipeline.ingest.download_video

        def _flaky_download(url, dest_dir):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise IngestError("simulated failure for first task only")
            return FAKE_VIDEO_PATH

        monkeypatch.setattr(pipeline.ingest, "download_video", _flaky_download)

        tasks = [_task(task_id="fails"), _task(task_id="succeeds")]
        deadline = time.monotonic() + 60.0

        results = asyncio.run(pipeline.run_all(tasks, _settings(), deadline))
        by_id = {r.task_id: r for r in results}

        assert by_id["fails"].captions["formal"] == _settings().fallback_caption
        assert by_id["succeeds"].captions["formal"] == "caption for formal"

    def test_tasks_not_started_before_deadline_are_skipped(self, monkeypatch):
        _patch_happy_path(monkeypatch)
        tasks = [_task(task_id=f"t{i}") for i in range(3)]
        already_passed_deadline = time.monotonic() - 1.0

        results = asyncio.run(pipeline.run_all(tasks, _settings(), already_passed_deadline))

        # None of the tasks should have been started; main.py's prefilled
        # fallback is responsible for covering them.
        assert results == []

    def test_run_all_never_raises_even_if_run_task_raises_unexpectedly(self, monkeypatch):
        _patch_happy_path(monkeypatch)

        async def _boom(task, settings, deadline=None):
            raise RuntimeError("simulated total run_task crash")

        monkeypatch.setattr(pipeline, "run_task", _boom)
        tasks = [_task(task_id="t1")]
        deadline = time.monotonic() + 60.0

        results = asyncio.run(pipeline.run_all(tasks, _settings(), deadline))

        assert results == []
