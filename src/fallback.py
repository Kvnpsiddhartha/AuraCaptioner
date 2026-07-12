"""The 'can't fail' layer.

Every guarantee the project depends on lives here: a valid TaskResult exists
for every task before any AI logic runs, missing styles are never left out,
and the final JSON written to disk always matches the documented shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.schemas import Style, Task, TaskResult


def prefill_results(tasks: list[Task], fallback_text: str) -> list[TaskResult]:
    """Build one TaskResult per task with every requested style mapped to
    `fallback_text`. Called immediately at startup, before any AI call, so
    there is always something safe to write.
    """
    return [
        TaskResult(
            task_id=task.task_id,
            captions={style: fallback_text for style in task.styles},
        )
        for task in tasks
    ]


def merge_result(base: TaskResult, produced: TaskResult) -> TaskResult:
    """Overlay `produced` captions onto `base`'s captions.

    `produced` values win per-style, but any style present in `base` and
    missing from `produced` falls back to `base`'s value — a style is never
    dropped just because the real pipeline didn't return it.
    """
    if base.task_id != produced.task_id:
        raise ValueError(
            f"task_id mismatch in merge_result: base={base.task_id!r} "
            f"produced={produced.task_id!r}"
        )
    merged: dict[Style, str] = dict(base.captions)
    for style, text in produced.captions.items():
        if text:
            merged[style] = text
    return TaskResult(task_id=base.task_id, captions=merged)


def validate_results(results: list[TaskResult], tasks: list[Task]) -> bool:
    """True only if every task_id from `tasks` appears exactly once in
    `results`, and every style requested for that task has a non-empty
    string value.
    """
    by_id: dict[str, TaskResult] = {}
    for result in results:
        if result.task_id in by_id:
            return False  # duplicate task_id
        by_id[result.task_id] = result

    for task in tasks:
        result = by_id.get(task.task_id)
        if result is None:
            return False  # task missing entirely
        for style in task.styles:
            text = result.captions.get(style)
            if not text or not text.strip():
                return False  # style missing or empty
    return True


def write_results(results: list[TaskResult], out_path: Path) -> None:
    """Write the bare JSON array (list of dicts), matching
    /output/results.json's documented shape exactly — NOT wrapped in an
    object.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump(mode="json") for result in results]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
