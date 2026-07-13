"""Pipeline orchestration: wires ingest -> sampling -> audio/ocr ->
grounding -> verification -> styling -> judge into one per-task chain, and
fans that out across all tasks under a wall-clock deadline.

This is the one file all the parallel Stage 1/Stage 2 modules converge
into. Every stage is wrapped in its own try/except so a failure at any
point degrades to the best available partial state rather than aborting
the whole task — per the project's core "fail INTO a fallback, never
crash" philosophy. `run_task` itself never raises: any unexpected failure
anywhere in the chain is caught and mapped to a fallback TaskResult.

Deadline behavior (documented per Prompt 5's requirement): `run_all` takes
a `deadline` (a `time.monotonic()` cutoff). Any task not yet STARTED
(including tasks still waiting on the concurrency semaphore) by the
deadline is skipped entirely and left for main.py's pre-filled fallback to
cover. Tasks already in flight are allowed to run to completion rather
than being forcibly cancelled — mid-flight cancellation risks leaving
partial downloads/temp files and, worse, a half-written result; letting an
in-flight task finish (or hit its own internal per-stage timeouts) is
simpler and safer, and main.py's overall process is still bounded by the
container's external runtime limit regardless.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from src import audio, grounding, ingest, judge, ocr, sampling, styling, verification
from src.config import Settings
from src.schemas import GroundingFacts, Style, Task, TaskResult

logger = logging.getLogger(__name__)

# Concurrency bound across tasks. Not exposed via Settings (the schema is
# frozen per PROJECT_CONTEXT) — sized conservatively by default to avoid
# hammering a single LLM/API provider with the full ~12-clip hidden test
# set at once. Prompt 9 hardening: overridable via MAX_CONCURRENT_TASKS
# env var so this can be tuned per-provider without a code change, while
# still falling back to a safe default if unset/invalid.
_DEFAULT_MAX_CONCURRENT_TASKS = 4


def _max_concurrent_tasks() -> int:
    raw = os.environ.get("MAX_CONCURRENT_TASKS")
    if not raw:
        return _DEFAULT_MAX_CONCURRENT_TASKS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "MAX_CONCURRENT_TASKS=%r is not an integer, using default %d",
            raw, _DEFAULT_MAX_CONCURRENT_TASKS,
        )
        return _DEFAULT_MAX_CONCURRENT_TASKS
    if value < 1:
        logger.warning(
            "MAX_CONCURRENT_TASKS=%r must be >= 1, using default %d",
            raw, _DEFAULT_MAX_CONCURRENT_TASKS,
        )
        return _DEFAULT_MAX_CONCURRENT_TASKS
    return value

# Lighter per-task time check (Prompt 7): if fewer than this many seconds
# remain before the overall deadline, skip the judge/regenerate step for
# this task and keep the pre-judge captions rather than risk blowing the
# runtime budget on a quality upgrade that was never required for
# correctness. judge_and_regenerate can roughly double/triple the LLM
# calls for weak styles, so this needs real margin, not just "> 0".
# Raised from 20 → 45 s: judge + optional regen can cost ~30-60 s per task;
# skipping it when fewer than 45 s remain is the safer tradeoff.
JUDGE_MIN_SECONDS_REQUIRED = 45.0


def _fallback_task_result(task: Task, settings: Settings) -> TaskResult:
    return TaskResult(
        task_id=task.task_id,
        captions={style: settings.fallback_caption for style in task.styles},
    )


async def run_task(
    task: Task,
    settings: Settings,
    deadline: float | None = None,
) -> TaskResult:
    """Run the full per-task chain and return a `TaskResult`.

    `deadline`, if given, is a `time.monotonic()` cutoff used only for the
    lighter per-task judge-skip check described in the module docstring —
    it does not abort earlier stages, which already have their own
    internal timeouts/retry budgets.

    Never raises: any exception anywhere in the chain (including one this
    function didn't anticipate) is caught at the outer boundary and mapped
    to a fallback TaskResult with every requested style set to
    `settings.fallback_caption`, matching the project's "fail INTO a
    fallback" contract.
    """
    try:
        return await _run_task_inner(task, settings, deadline)
    except Exception as exc:  # noqa: BLE001 - final outer safety net
        logger.warning(
            "task_id=%s: unexpected exception escaped the pipeline, using fallback: %s",
            task.task_id, exc,
        )
        return _fallback_task_result(task, settings)


async def _run_task_inner(
    task: Task,
    settings: Settings,
    deadline: float | None,
) -> TaskResult:
    fallback_result = _fallback_task_result(task, settings)

    workdir = Path(tempfile.mkdtemp(prefix=f"vcap_{task.task_id}_"))
    try:
        # --- Stage: ingest -------------------------------------------------
        try:
            video_path = await asyncio.to_thread(ingest.download_video, task.video_url, workdir)
            duration = await asyncio.to_thread(ingest.probe_duration, video_path)
            video_path = await asyncio.to_thread(ingest.downscale, video_path)
        except ingest.IngestError as exc:
            logger.warning("task_id=%s: ingest failed, using fallback: %s", task.task_id, exc)
            return fallback_result
        except Exception as exc:  # defense-in-depth: ingest.py should only raise IngestError
            logger.warning("task_id=%s: unexpected ingest failure, using fallback: %s", task.task_id, exc)
            return fallback_result

        # --- Stage: frame sampling -----------------------------------------
        frames_dir = workdir / "frames"
        n_frames = sampling.frame_count_for_duration(
            duration, settings.frames_min, settings.frames_max
        )
        frames: list[Path] = []
        try:
            frames = await asyncio.to_thread(sampling.sample_keyframes, video_path, n_frames, frames_dir)
        except Exception as exc:
            logger.warning("task_id=%s: sample_keyframes failed: %s", task.task_id, exc)
        if not frames:
            try:
                frames = await asyncio.to_thread(sampling.sample_uniform, video_path, n_frames, frames_dir)
            except Exception as exc:
                logger.warning("task_id=%s: sample_uniform fallback also failed: %s", task.task_id, exc)

        if not frames:
            logger.warning("task_id=%s: no frames could be extracted, using fallback", task.task_id)
            return fallback_result

        # --- Stage: audio + OCR (independent, run concurrently) ------------
        transcript: str | None = None
        ocr_text: list[str] = []
        audio_result, ocr_result = await asyncio.gather(
            asyncio.to_thread(audio.transcribe, video_path),
            asyncio.to_thread(ocr.extract_text, frames),
            return_exceptions=True,
        )
        if isinstance(audio_result, BaseException):
            logger.warning("task_id=%s: audio.transcribe raised unexpectedly: %s", task.task_id, audio_result)
        else:
            transcript = audio_result
        if isinstance(ocr_result, BaseException):
            logger.warning("task_id=%s: ocr.extract_text raised unexpectedly: %s", task.task_id, ocr_result)
        else:
            ocr_text = ocr_result

        # --- Stage: grounding ------------------------------------------------
        try:
            facts: GroundingFacts = await grounding.ground_video(frames, transcript, ocr_text, settings)
        except grounding.GroundingError as exc:
            logger.warning("task_id=%s: grounding failed, using fallback: %s", task.task_id, exc)
            return fallback_result
        except Exception as exc:  # defense-in-depth
            logger.warning("task_id=%s: unexpected grounding failure, using fallback: %s", task.task_id, exc)
            return fallback_result

        # --- Stage: verification ---------------------------------------------
        # verify_facts already never raises (internally passes facts
        # through unmodified on failure), but wrap anyway per the
        # project's "every stage wrapped" rule.
        # Gated on settings.enable_verification (default False) because the
        # 3 sequential LLM sub-steps add ~90-135 s per task — too costly
        # under a 10-minute container budget with multiple clips.
        cleaned_facts = facts
        if settings.enable_verification:
            try:
                verification_result = await verification.verify_facts(facts, frames, settings)
                cleaned_facts = verification_result.cleaned_facts
            except Exception as exc:
                logger.warning(
                    "task_id=%s: verification raised unexpectedly, using ungrounded facts: %s",
                    task.task_id, exc,
                )
                cleaned_facts = facts
        else:
            logger.debug("task_id=%s: verification skipped (ENABLE_VERIFICATION=false)", task.task_id)

        # --- Stage: styling ----------------------------------------------------
        try:
            captions: dict[Style, str] = await styling.style_all(cleaned_facts, task.styles, settings)
        except Exception as exc:
            logger.warning("task_id=%s: styling raised unexpectedly, using fallback: %s", task.task_id, exc)
            captions = {style: settings.fallback_caption for style in task.styles}

        # --- Stage: judge + regenerate (Prompt 7) -------------------------
        # A quality upgrade, never a blocker: any failure here falls
        # through with the pre-judge captions unchanged. Also skipped
        # outright if too little time remains before the overall
        # deadline, since this can roughly double/triple the LLM calls
        # for weak styles.
        # Gated on settings.enable_judge (default True); set ENABLE_JUDGE=false
        # to skip entirely when the runtime budget is dangerously tight.
        time_remaining = None if deadline is None else deadline - time.monotonic()
        judge_time_ok = time_remaining is None or time_remaining >= JUDGE_MIN_SECONDS_REQUIRED
        if not settings.enable_judge:
            logger.debug("task_id=%s: judge skipped (ENABLE_JUDGE=false)", task.task_id)
        elif not judge_time_ok:
            logger.info(
                "task_id=%s: skipping judge step, only %.1fs left before deadline",
                task.task_id, time_remaining,
            )
        else:
            try:
                captions = await judge.judge_and_regenerate(
                    facts=cleaned_facts,
                    captions=captions,
                    settings=settings,
                    regenerate_fn=lambda f, s, cfg: styling.style_caption(f, s, cfg),
                    max_retries=settings.max_self_judge_retries,
                )
            except Exception as exc:
                logger.warning(
                    "task_id=%s: judge_and_regenerate raised unexpectedly, "
                    "keeping pre-judge captions: %s",
                    task.task_id, exc,
                )

        return TaskResult(task_id=task.task_id, captions=captions)
    finally:
        await asyncio.to_thread(shutil.rmtree, workdir, ignore_errors=True)


async def run_all(
    tasks: list[Task],
    settings: Settings,
    deadline: float,
) -> list[TaskResult]:
    """Run `run_task` concurrently across `tasks`, bounded by
    MAX_CONCURRENT_TASKS and `deadline` (a `time.monotonic()` cutoff).

    See the module docstring for the documented deadline-handling
    contract. Returns only the results for tasks that actually ran;
    skipped/failed tasks are simply absent from the returned list, and
    main.py's merge-over-prefilled-fallback logic covers the gap — this
    function never raises and never returns a partial/malformed
    TaskResult.
    """
    semaphore = asyncio.Semaphore(_max_concurrent_tasks())

    async def _gated(task: Task) -> TaskResult | None:
        if time.monotonic() >= deadline:
            logger.warning(
                "task_id=%s: skipped, deadline already passed before start", task.task_id
            )
            return None
        async with semaphore:
            if time.monotonic() >= deadline:
                logger.warning(
                    "task_id=%s: skipped, deadline passed while waiting for a "
                    "concurrency slot", task.task_id,
                )
                return None
            try:
                return await run_task(task, settings, deadline=deadline)
            except Exception as exc:  # run_task shouldn't raise, but stay defensive
                logger.warning(
                    "task_id=%s: run_task raised unexpectedly at the top level: %s",
                    task.task_id, exc,
                )
                return None

    gathered = await asyncio.gather(*(_gated(task) for task in tasks))
    return [result for result in gathered if result is not None]


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run the video captioning pipeline on a single video.")
    parser.add_argument("--video-path", required=True, help="Path or URL to the video file.")
    parser.add_argument("--styles", nargs="+", default=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
                        help="Styles to generate captions for.")
    args = parser.parse_args()
    
    async def main():
        # Load .env file manually if it exists
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

        # Configure root logger to output to console
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        from src.config import load_settings
        from src.schemas import Task
        
        settings = load_settings()
        task = Task(
            task_id="cli_task",
            video_url=args.video_path,
            styles=args.styles
        )
        print(f"Running pipeline on: {args.video_path}")
        result = await run_task(task, settings)
        print("\n=== Generation Results ===")
        print(json.dumps(result.model_dump(), indent=2))

    asyncio.run(main())

