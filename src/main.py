"""Entrypoint for the video captioning agent.

Core contract (holds regardless of pipeline complexity): this file must
produce a valid, complete /output/results.json and exit 0, EVEN IF every
downstream AI call fails. Prompt 5 wires in the real pipeline
(src/pipeline.py); this module never calls it in a way that can weaken
that guarantee — the pre-fill / merge / validate / write chain built in
Phase 0 stays exactly as strict as it always was.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

from src import pipeline
from src.config import Settings, SettingsError, load_settings
from src.fallback import merge_result, prefill_results, validate_results, write_results
from src.schemas import Task, TaskResult

INPUT_PATH = Path(os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json"))


def _read_tasks(path: Path) -> list[Task]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("tasks.json must contain a JSON array of task objects")
    return [Task.model_validate(item) for item in data]


def main() -> int:
    # Load .env file manually if it exists in the app root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

    # Catch everything here: an unreadable /input/tasks.json is the one
    # scenario where we may have nothing at all to write. Even then, prefer
    # writing an empty-but-valid [] over a non-zero exit / crash.
    try:
        settings = load_settings()
    except SettingsError as exc:
        print(f"[main] FATAL: configuration error: {exc}", file=sys.stderr)
        return 1

    # Emit a startup marker so the AMD test harness (-ReadyPattern 'healthy|ready')
    # can confirm the container is alive and measure startup latency.
    print("[READY] AuraCaptioner started", flush=True)

    try:
        tasks = _read_tasks(INPUT_PATH)
    except Exception:
        print(f"[main] FATAL: could not read/parse {INPUT_PATH}:", file=sys.stderr)
        traceback.print_exc()
        try:
            write_results([], OUTPUT_PATH)
        except Exception:
            print("[main] could not even write an empty results.json", file=sys.stderr)
        return 0  # still prefer exit 0 with an empty (valid) array over crashing

    # Pre-fill fallback results BEFORE anything else runs, so there is
    # always something safe to write no matter what happens downstream.
    base_results = prefill_results(tasks, settings.fallback_caption)

    deadline = time.monotonic() + settings.max_runtime_seconds

    produced_results: list[TaskResult] = base_results
    try:
        produced_results = asyncio.run(pipeline.run_all(tasks, settings, deadline))
    except Exception:
        print("[main] pipeline raised — falling back to pre-filled results:", file=sys.stderr)
        traceback.print_exc()
        produced_results = base_results

    # Merge produced results over the pre-filled base so no style is ever
    # dropped, even if the pipeline only partially completed.
    produced_by_id = {result.task_id: result for result in produced_results}
    merged_results = [
        merge_result(base, produced_by_id[base.task_id])
        if base.task_id in produced_by_id
        else base
        for base in base_results
    ]

    if not validate_results(merged_results, tasks):
        print(
            "[main] merged results failed validation — writing pure fallback instead",
            file=sys.stderr,
        )
        merged_results = base_results

    try:
        write_results(merged_results, OUTPUT_PATH)
    except Exception:
        print(f"[main] FATAL: could not write {OUTPUT_PATH}:", file=sys.stderr)
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
