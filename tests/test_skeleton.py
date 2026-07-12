"""Phase 0 skeleton tests.

These prove the single most important guarantee in the project: no matter
what happens in the real pipeline, /output/results.json ends up valid and
complete, and the process exits 0.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import main as main_module
from src.fallback import prefill_results, validate_results
from src.schemas import Task, TaskResult

# These tests exercise the "can't fail" plumbing (prefill -> pipeline ->
# merge -> validate -> write), not the real AI pipeline stages themselves
# (that's covered end-to-end in tests/test_pipeline.py with mocked
# ingest/grounding/etc). Default pipeline.run_all to a fast, offline,
# deterministic stub that just echoes the fallback caption back, so these
# stay fast and don't depend on network/model access.


async def _fast_echo_run_all(tasks, settings, deadline):
    return prefill_results(tasks, settings.fallback_caption)


@pytest.fixture(autouse=True)
def _stub_pipeline(monkeypatch):
    monkeypatch.setattr(main_module.pipeline, "run_all", _fast_echo_run_all)

FIXTURE_TASKS = [
    {
        "task_id": "v1",
        "video_url": "https://storage.example.com/clips/clip1.mp4",
        "styles": ["formal", "sarcastic"],
    },
    {
        "task_id": "v2",
        "video_url": "https://storage.example.com/clips/clip2.mp4",
        "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
    },
]


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


def _tasks() -> list[Task]:
    return [Task.model_validate(item) for item in FIXTURE_TASKS]


def test_noop_run_produces_valid_complete_output(env_and_paths):
    tasks_path, output_path = env_and_paths

    exit_code = main_module.main()

    assert exit_code == 0
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)  # bare array, not wrapped in an object

    results = [TaskResult.model_validate(item) for item in data]
    assert validate_results(results, _tasks())


def test_pipeline_failure_still_produces_valid_complete_output(env_and_paths):
    tasks_path, output_path = env_and_paths

    async def _boom(tasks, settings, deadline):
        raise RuntimeError("simulated pipeline crash")

    with mock.patch.object(main_module.pipeline, "run_all", _boom):
        exit_code = main_module.main()

    assert exit_code == 0
    assert output_path.exists()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    results = [TaskResult.model_validate(item) for item in data]
    assert validate_results(results, _tasks())

    # Every value should be exactly the fallback caption since the
    # pipeline never got a chance to run.
    for result in results:
        for text in result.captions.values():
            assert text == "A short video clip."


def test_every_requested_style_present_for_every_task(env_and_paths):
    tasks_path, output_path = env_and_paths

    main_module.main()

    data = json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {item["task_id"]: item for item in data}

    for task_dict in FIXTURE_TASKS:
        result = by_id[task_dict["task_id"]]
        for style in task_dict["styles"]:
            assert style in result["captions"]
            assert result["captions"][style].strip()


def test_unreadable_tasks_file_still_exits_zero(tmp_path, monkeypatch):
    bad_tasks_path = tmp_path / "tasks.json"
    bad_tasks_path.write_text("{not valid json", encoding="utf-8")
    output_path = tmp_path / "results.json"

    monkeypatch.setenv("GROUNDING_MODEL", "test-grounding-model")
    monkeypatch.setenv("STYLING_MODEL", "test-styling-model")
    monkeypatch.setenv("JUDGE_MODEL", "test-judge-model")
    monkeypatch.setattr(main_module, "INPUT_PATH", bad_tasks_path)
    monkeypatch.setattr(main_module, "OUTPUT_PATH", output_path)

    exit_code = main_module.main()

    assert exit_code == 0
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data == []


def test_noop_path_is_fast(env_and_paths):
    start = time.monotonic()
    main_module.main()
    elapsed = time.monotonic() - start
    # Sanity check the watchdog/deadline machinery doesn't itself add delay.
    assert elapsed < 5.0
