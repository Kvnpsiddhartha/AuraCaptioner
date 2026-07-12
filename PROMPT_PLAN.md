# Video Captioning Agent — Prompt Plan & Execution Order

Paste `PROJECT_CONTEXT` at the top of **every** prompt below (it's the shared
mental model — schemas, layout, constraints). Then paste the numbered prompt
body for the step you're on.

---

## PROJECT_CONTEXT (paste with every prompt, unmodified)

```
You are building a video-captioning agent for a hackathon with hard constraints:
- exit code 0 on success, max runtime 10 min, image ≤10GB compressed
- input: /input/tasks.json → list of {task_id, video_url, styles: [formal|sarcastic|humorous_tech|humorous_non_tech]}
- output: /output/results.json → list of {task_id, captions: {style: caption_text}}
  (bare JSON array, NOT wrapped in an object)
- missing a requested style for a clip = zero for that clip
- malformed JSON output = zero for the whole run
- philosophy: fail INTO a pre-filled fallback, never crash. Every stage is
  wrapped so a failure degrades gracefully instead of losing the run.

Stack: Python 3.11, async where calls are I/O-bound (LLM/API calls), pydantic
v2 for all schemas, ffmpeg via subprocess for video work, structured/schema-
enforced output for every LLM call (never free-text parsing).

Repo layout (do not deviate from these paths — later prompts assume them):
  src/schemas.py
  src/config.py
  src/fallback.py
  src/main.py
  src/ingest.py
  src/sampling.py
  src/audio.py
  src/ocr.py
  src/grounding.py
  src/verification.py
  src/styling.py
  src/judge.py
  src/pipeline.py
  Dockerfile
  requirements.txt
  tests/

Canonical schemas (src/schemas.py — treat as frozen once written in Prompt 1;
later prompts import from here, they do not redefine these):

  Style = Literal["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

  class Task(BaseModel):
      task_id: str
      video_url: str
      styles: list[Style]

  class GroundingFacts(BaseModel):
      subjects: list[str]
      actions: list[str]
      setting: str
      mood: str
      on_screen_text: list[str] = []
      audible_speech: str | None = None
      notable_sounds: list[str] = []

  class VerificationResult(BaseModel):
      original_facts: GroundingFacts
      verification_questions: list[str]
      verification_answers: list[str]
      cleaned_facts: GroundingFacts
      dropped_claims: list[str] = []

  class StyledCaption(BaseModel):
      style: Style
      text: str

  class JudgeScore(BaseModel):
      style: Style
      accuracy: float          # 0..1
      style_match: float       # 0..1
      notes: str | None = None

  class TaskResult(BaseModel):
      task_id: str
      captions: dict[Style, str]

Canonical config (src/config.py — Settings is imported everywhere; do not
re-declare fields, only read them):

  class Settings(BaseModel):
      max_runtime_seconds: int = 570   # 9.5 min, safety margin under 10:00
      frames_min: int = 8
      frames_max: int = 20
      grounding_model: str
      styling_model: str
      judge_model: str
      fallback_caption: str = "A short video clip."
      max_self_judge_retries: int = 2

Global rule: every module-level function that calls an external process,
network, or model MUST catch its own exceptions and return a safe default
(never raise past its own boundary) — pipeline.py is the only place allowed
to log-and-continue at the task level, but each module should already be
defensive on its own.
```

---

## Execution Graph

```
Prompt 1 (schemas + config)
        │
Prompt 2 (Phase 0 skeleton: fallback.py, main.py, Dockerfile, no-op run)
        │
        ├── Prompt 3A (ingest.py)      ─┐
        ├── Prompt 3B (sampling.py)    ─┼─ PARALLEL (different files, only
        ├── Prompt 3C (audio.py)       ─┤   depend on schemas.py/config.py)
        └── Prompt 3D (ocr.py)         ─┘
        │
        ├── Prompt 4A (grounding.py)   ─┐
        ├── Prompt 4B (verification.py)─┼─ PARALLEL (different files, only
        └── Prompt 4C (styling.py)     ─┘   depend on schemas.py)
        │
Prompt 5 (pipeline.py — wires everything, SEQUENTIAL, single file)
        │
        ├── Prompt 6A (judge.py)              ─┐ PARALLEL (different files,
        └── Prompt 6B (style_rubrics content   ─┘  only depend on schemas.py)
             inside styling.py — see note)
        │
Prompt 7 (integrate judge + rubrics into pipeline.py — SEQUENTIAL, edits
          pipeline.py, must come after 6A/6B)
        │
        ├── Prompt 8A (native_video.py, optional)   ─┐ PARALLEL, isolated
        └── Prompt 8B (finetune/ scaffold, optional) ─┘ experimental files
        │
Prompt 9 (hardening pass — concurrency, provider fallback, size/runtime
          checks — SEQUENTIAL, edits main.py + pipeline.py)
```

Note on 6B: since `STYLE_RUBRICS` lives inside `styling.py` (created in 4C),
Prompt 6B is phrased as "add rubric constants to styling.py" — if you truly
want it parallel-safe, put rubrics in a new `src/rubrics.py` instead and have
styling.py import from it. Both options are given below; pick one.

---

## Prompt 1 — Schemas & Config (sequential, first)

```
Files to create: src/schemas.py, src/config.py, requirements.txt

Nothing exists yet — this is the foundation every other module imports from.

Build src/schemas.py with exactly these pydantic v2 models (copy verbatim
from PROJECT_CONTEXT's "Canonical schemas" block above — do not add fields
that aren't there, other modules will assume this exact shape):
  Style, Task, GroundingFacts, VerificationResult, StyledCaption, JudgeScore,
  TaskResult

Build src/config.py with the Settings model from PROJECT_CONTEXT, plus:
  def load_settings() -> Settings
    - reads model names from env vars GROUNDING_MODEL, STYLING_MODEL,
      JUDGE_MODEL (required, raise a clear error at startup if missing —
      this is the one place a hard failure at boot is acceptable, before
      any task processing starts)
    - everything else has the defaults shown

Build requirements.txt with: pydantic>=2, and leave a comment block listing
what later prompts will add (ffmpeg-python or subprocess-only, an LLM SDK of
your choice, faster-whisper or groq, an OCR lib) — don't pin those yet,
later prompts will append as needed.

Verification checklist:
[ ] `python -c "from src.schemas import Task, GroundingFacts, TaskResult"` succeeds
[ ] `python -c "from src.config import load_settings"` succeeds
[ ] Task, GroundingFacts, StyledCaption, JudgeScore, TaskResult all round-trip
    through .model_dump_json() / .model_validate_json()
[ ] Settings() without required model env vars raises a clear, caught error —
    not a bare pydantic ValidationError traceback
```

## Prompt 2 — Phase 0 Skeleton (sequential, depends on Prompt 1)

```
Previously built: src/schemas.py (Task, TaskResult, Style, etc.),
src/config.py (Settings, load_settings()).

Files to create: src/fallback.py, src/main.py, Dockerfile, tests/test_skeleton.py

This is the "can't fail" layer — it must work end-to-end with a NO-OP
pipeline before any real AI logic exists, per the project's Phase 0 rule.

Build src/fallback.py:
  def prefill_results(tasks: list[Task], fallback_text: str) -> list[TaskResult]
      - returns one TaskResult per task, with every requested style in
        task.styles mapped to fallback_text
  def merge_result(base: TaskResult, produced: TaskResult) -> TaskResult
      - produced values override base per-style, but any style missing from
        produced falls back to base's value (never drop a style)
  def validate_results(results: list[TaskResult], tasks: list[Task]) -> bool
      - True only if every task_id from tasks appears exactly once in results
        AND every requested style for that task has a non-empty string value
  def write_results(results: list[TaskResult], out_path: Path) -> None
      - writes the bare JSON array (list of dicts via model_dump), NOT a
        wrapped object — this must match /output/results.json's documented
        shape exactly

Build src/main.py:
  def main() -> int
      - reads /input/tasks.json, parses into list[Task]
      - calls fallback.prefill_results() immediately, BEFORE anything else,
        so there is always something to write
      - starts a wall-clock deadline = now + settings.max_runtime_seconds
      - for now (Phase 0), calls a stub `run_all_noop(tasks) -> list[TaskResult]`
        that just echoes the prefilled fallback results back (proves the
        plumbing works with zero AI calls)
      - merges results over the prefilled base via fallback.merge_result
      - validates via fallback.validate_results; if invalid, writes the pure
        prefilled fallback instead
      - writes to /output/results.json
      - returns 0 always (catch everything at this top level; only a
        completely unreadable /input/tasks.json should risk non-zero, and
        even then prefer writing an empty-but-valid [] over crashing)
  if __name__ == "__main__": sys.exit(main())

Build Dockerfile:
  - FROM python:3.11-slim (or similar minimal base)
  - copies src/, installs requirements.txt
  - CMD ["python", "-m", "src.main"]
  - confirms /input and /output are the expected mount points (document this
    assumption in a comment, don't hardcode paths that break if the harness
    mounts elsewhere — read from env var with these as defaults)

Build tests/test_skeleton.py:
  - a fixture tasks.json with 2 tasks, one with 2 styles, one with 4 styles
  - test that main() produces valid JSON with every task_id/style present
  - test that if run_all_noop is monkeypatched to raise, output still has
    every task_id/style present (fallback path proven)

Verification checklist:
[ ] `docker build` succeeds and image exists
[ ] running the container against a fixture /input/tasks.json produces
    /output/results.json that is valid JSON and matches the documented shape
[ ] exit code is 0 in the no-op case
[ ] exit code is 0 even when run_all_noop is forced to raise (test this
    explicitly — this is the single most important guarantee in the project)
[ ] every style in every task's `styles` list appears in the output for
    that task_id, no exceptions
[ ] total run time for the no-op path is trivially fast (sanity check the
    watchdog logic doesn't itself introduce delay)
```

## Prompt 3A — Ingest (parallel with 3B/3C/3D)

```
Previously built: src/schemas.py, src/config.py (Settings), src/fallback.py,
src/main.py (Phase 0 skeleton, currently calling a no-op pipeline).

Files to create: src/ingest.py only. Do not touch main.py, fallback.py, or
any other module — this must be safely parallel with sampling.py, audio.py,
ocr.py which are being built at the same time against the same base.

Build src/ingest.py:
  def download_video(url: str, dest_dir: Path) -> Path
      - downloads video_url to dest_dir, returns local path
      - must have a timeout and raise a specific IngestError on failure
        (define `class IngestError(Exception)` in this file) so pipeline.py
        can catch it distinctly later
  def probe_duration(path: Path) -> float
      - uses ffprobe via subprocess, returns duration in seconds
      - raises IngestError on unparseable output
  def downscale(path: Path, max_height: int = 480) -> Path
      - ffmpeg subprocess call, returns path to downscaled copy
      - purpose: keep frame extraction and upload payloads small/fast

Verification checklist:
[ ] download_video raises IngestError (not a raw requests/urllib exception)
    on a bad URL, with a message identifying the task
[ ] probe_duration returns a float > 0 for a known-good test clip
[ ] downscale output file exists and is smaller in resolution than input
    (verify via ffprobe on the output)
[ ] no function in this file leaks an unhandled exception type outside
    IngestError
```

## Prompt 3B — Frame Sampling (parallel with 3A/3C/3D)

```
Previously built: same base as 3A (schemas.py, config.py, fallback.py,
main.py skeleton). ingest.py is being built in parallel — do not import
from it; take a `video_path: Path` as input instead.

Files to create: src/sampling.py only.

Build src/sampling.py:
  def frame_count_for_duration(duration_s: float, fmin: int = 8, fmax: int = 20) -> int
      - scales frame count with clip length, clamped to [fmin, fmax]
      - simple linear or sqrt scaling is fine; document the formula in a
        comment since Phase 2 may tune it
  def sample_uniform(video_path: Path, n: int, out_dir: Path) -> list[Path]
      - fixed-interval ffmpeg frame extraction, n frames, returns sorted
        list of frame image paths (used for Phase 1 MVP)
  def sample_keyframes(video_path: Path, n: int, out_dir: Path) -> list[Path]
      - scene-change/keyframe-based extraction (e.g. ffmpeg `select='gt(scene,...)'`
        filter, or an equivalent detection approach), falling back to
        sample_uniform if fewer than n scene-change frames are found
      - returns sorted list of frame image paths (used from Phase 2 onward)

Verification checklist:
[ ] frame_count_for_duration(30) and frame_count_for_duration(120) both fall
    within [fmin, fmax] and the longer clip gets >= frames vs the shorter one
[ ] sample_uniform returns exactly n paths (or fewer only if video is
    shorter than n distinct frames), all files exist and are valid images
[ ] sample_keyframes falls back to sample_uniform's behavior gracefully on
    a static/no-scene-change test clip instead of returning too few frames
[ ] both functions clean up or namespace their output under out_dir so
    concurrent tasks don't collide on filenames
```

## Prompt 3C — Audio (VAD + Whisper) (parallel with 3A/3B/3D)

```
Previously built: same base as 3A. ingest.py/sampling.py are being built in
parallel — take `video_path: Path` as input, don't import from them.

Files to create: src/audio.py only.

Build src/audio.py:
  def has_speech(video_path: Path) -> bool
      - lightweight voice-activity check (energy-based or a small VAD lib)
      - purpose: gate the expensive transcription step, per the plan's
        "don't waste time on silent clips" finding — this must be cheap
      - return False (not raise) on any failure — silence is the safe default
  def transcribe(video_path: Path) -> str | None
      - calls has_speech() first internally; returns None immediately if
        no speech detected (do not call the transcription model)
      - otherwise runs Whisper (local or hosted, your choice per model
        notes) and returns the transcript text, or None on failure

Verification checklist:
[ ] has_speech returns False on a silent/near-silent test clip
[ ] has_speech returns True on a clip with clear speech
[ ] transcribe returns None (not an exception) when has_speech is False,
    and does not make a network/model call in that path (verify via a
    mock/spy in the test)
[ ] transcribe returns None rather than raising if the model call itself
    fails
```

## Prompt 3D — OCR (parallel with 3A/3B/3C)

```
Previously built: same base as 3A. sampling.py is being built in parallel —
this module takes `frames: list[Path]` as input (plain image paths), it
does not import sampling.py.

Files to create: src/ocr.py only.

Build src/ocr.py:
  def has_probable_text(frames: list[Path]) -> bool
      - cheap heuristic pass (e.g. edge-density/contrast check, or a fast
        text-region detector) to decide if full OCR is worth running
      - return False on any failure — safe default, skip OCR
  def extract_text(frames: list[Path]) -> list[str]
      - calls has_probable_text() first; returns [] immediately if False
      - otherwise runs OCR (e.g. RapidOCR) across the frames, returns
        deduplicated, cleaned text strings found

Verification checklist:
[ ] has_probable_text returns False on frames with no text (plain nature/
    animal footage) — avoids wasted OCR calls
[ ] has_probable_text returns True on frames with clear on-screen text/signage
[ ] extract_text returns [] without invoking OCR when has_probable_text is
    False (verify via mock/spy)
[ ] extract_text output has no exact duplicate strings and no empty strings
```

## Prompt 4A — Grounding (Stage 1) (parallel with 4B/4C)

```
Previously built: src/schemas.py (GroundingFacts), src/config.py (Settings).
ingest.py, sampling.py, audio.py, ocr.py now exist (Prompts 3A–3D) and
produce: list[Path] frames, str|None transcript, list[str] ocr_text — this
module CONSUMES those shapes as plain function arguments, it does not import
those modules directly (keeps it swappable/testable in isolation).
verification.py and styling.py are being built in parallel against the same
GroundingFacts schema — do not modify schemas.py.

Files to create: src/grounding.py only.

Build src/grounding.py:
  class GroundingError(Exception): ...
  async def ground_video(
      frames: list[Path],
      transcript: str | None,
      ocr_text: list[str],
      settings: Settings,
  ) -> GroundingFacts
      - sends frames (+ transcript/ocr context if present) to
        settings.grounding_model via a VLM call
      - REQUIRES schema-enforced/structured output that maps directly onto
        GroundingFacts (use tool-calling/JSON-schema forcing, not free-text
        parsing — this is a hard requirement from the research notes)
      - on any failure (API error, schema validation failure), raise
        GroundingError with task context in the message rather than
        returning a partially-filled or guessed GroundingFacts

Verification checklist:
[ ] ground_video returns a valid GroundingFacts instance for a known test
    clip (subjects/actions/setting non-empty)
[ ] on a clip with no speech (transcript=None) and no OCR text (ocr_text=[]),
    it still returns a valid GroundingFacts using visual info alone
[ ] a forced malformed-response from the model surfaces as GroundingError,
    not an unhandled JSON/pydantic exception
[ ] output contains no fields/claims not visible in the actual test frames
    (spot-check manually against 2-3 clips)
```

## Prompt 4B — Verification / CoVe (parallel with 4A/4C)

```
Previously built: src/schemas.py (GroundingFacts, VerificationResult),
src/config.py (Settings). This module takes a GroundingFacts object as
input (as if grounding.py already produced it) — do not import grounding.py,
stay decoupled so this is independently testable.

Files to create: src/verification.py only.

Build src/verification.py:
  class VerificationError(Exception): ...
  async def verify_facts(
      facts: GroundingFacts,
      frames: list[Path],
      settings: Settings,
  ) -> VerificationResult
      - implements Chain-of-Verification: (1) generate verification
        questions about `facts`, (2) answer them independently against
        `frames`, (3) produce cleaned_facts with any unsupported claims
        removed and listed in dropped_claims
      - each of the 3 sub-steps should be its own schema-enforced call (or
        clearly separated internal helper functions) — don't collapse into
        one prompt, the plan's research basis is specifically the multi-step
        version
      - on failure at any sub-step, catch it and return a VerificationResult
        where cleaned_facts == original_facts and dropped_claims == []
        (i.e. verification failing should not block the pipeline — grounding
        facts pass through unmodified rather than the task failing)

Verification checklist:
[ ] verify_facts on a deliberately fabricated GroundingFacts (e.g. an
    animal that isn't in the test frames) results in that claim appearing
    in dropped_claims and absent from cleaned_facts
[ ] verify_facts on accurate GroundingFacts returns cleaned_facts
    equivalent to the input (no false-positive drops)
[ ] a forced failure in the verification model call falls back to
    cleaned_facts == original facts, never raises out of this function
[ ] verification_questions and verification_answers are non-empty when the
    pass succeeds (proof the multi-step CoVe actually ran, not a shortcut)
```

## Prompt 4C — Styling (Stage 2) (parallel with 4A/4B)

```
Previously built: src/schemas.py (GroundingFacts, StyledCaption, Style),
src/config.py (Settings). Takes a GroundingFacts object as input — do not
import grounding.py or verification.py, stay decoupled.

Files to create: src/styling.py (and, if you're choosing the "rubrics in a
separate file" option, src/rubrics.py — otherwise rubric constants live in
this file; see note under Prompt 6B).

Build src/styling.py:
  STYLE_RUBRICS: dict[Style, str]
      - a short explicit rubric string per style with a positive instruction
        and a negative constraint, e.g.:
          "sarcastic": rewards dry understatement and irony; forbid
              exclamation marks and enthusiastic adjectives
          "humorous_non_tech": rewards accessible, everyday humor; forbid
              jargon/technical terminology entirely
          "humorous_tech": rewards tech-culture references/wordplay;
              assumes a technical audience
          "formal": rewards precise, neutral, structured phrasing; forbid
              slang, contractions, humor
        (placeholder text is fine for now — Phase 3 refines these, but the
        dict shape and per-style negative constraint must exist now since
        other prompts assume this dict's keys)
  async def style_caption(
      facts: GroundingFacts, style: Style, settings: Settings
  ) -> StyledCaption
      - single schema-enforced call using STYLE_RUBRICS[style], grounded
        ONLY in `facts` (never re-derive facts from the style prompt)
      - keep output SHORT — explicitly instruct brevity in the prompt to
        guard against LLM-judge verbosity bias (see PROJECT_CONTEXT)
      - on failure, return StyledCaption(style=style, text=settings.fallback_caption)
        rather than raising — styling failures are per-style and should
        never block sibling styles
  async def style_all(
      facts: GroundingFacts, styles: list[Style], settings: Settings
  ) -> dict[Style, str]
      - runs style_caption concurrently (asyncio.gather) for all requested
        styles, returns {style: text}, guaranteed to have every requested
        style as a key (using style_caption's fallback behavior)

Verification checklist:
[ ] style_all(facts, ["formal","sarcastic","humorous_tech","humorous_non_tech"], settings)
    returns a dict with all 4 keys populated with non-empty strings
[ ] captions for different styles on the same facts are visibly different
    in tone (manual spot check)
[ ] humorous_non_tech output contains no jargon terms present in
    humorous_tech's output for the same facts
[ ] a forced failure on exactly one style (e.g. mock style_caption to raise
    for "sarcastic") still returns all 4 keys, with "sarcastic" falling
    back to settings.fallback_caption
[ ] caption lengths are short (spot-check word counts — no style should
    balloon relative to the others)
```

## Prompt 5 — Pipeline Orchestration (sequential, single file)

```
Previously built (all now exist and are stable):
  src/ingest.py       → download_video, probe_duration, downscale, IngestError
  src/sampling.py     → frame_count_for_duration, sample_uniform, sample_keyframes
  src/audio.py        → has_speech, transcribe
  src/ocr.py          → has_probable_text, extract_text
  src/grounding.py    → ground_video(frames, transcript, ocr_text, settings) -> GroundingFacts, GroundingError
  src/verification.py → verify_facts(facts, frames, settings) -> VerificationResult
  src/styling.py      → style_all(facts, styles, settings) -> dict[Style, str]
  src/fallback.py     → prefill_results, merge_result, validate_results, write_results
  src/schemas.py      → Task, TaskResult, GroundingFacts, etc.
  src/config.py       → Settings, load_settings

This prompt wires them together. This is the one file all the parallel
modules converge into — do it as a single sequential step, not parallel.

Files to create: src/pipeline.py. Files to edit: src/main.py (swap the
Phase 0 no-op call for the real pipeline).

Build src/pipeline.py:
  async def run_task(task: Task, settings: Settings) -> TaskResult
      - full chain: ingest.download_video → ingest.probe_duration →
        ingest.downscale → sampling.frame_count_for_duration →
        sampling.sample_keyframes (fallback sample_uniform is internal to
        that function already) → audio.transcribe → ocr.extract_text →
        grounding.ground_video → verification.verify_facts →
        styling.style_all(cleaned_facts, task.styles, settings)
      - wraps EACH stage in its own try/except; on failure at any stage,
        continue with the best available partial state rather than aborting
        the whole task (e.g. grounding failure → skip straight to a
        fallback TaskResult for this task; verification failure → already
        handled inside verify_facts, facts pass through)
      - returns TaskResult(task_id=task.task_id, captions=<style_all result>)
      - if grounding itself fails (GroundingError), return
        TaskResult with every task.styles key set to settings.fallback_caption
  async def run_all(tasks: list[Task], settings: Settings, deadline: float) -> list[TaskResult]
      - runs run_task concurrently across tasks (asyncio.gather or a
        semaphore-bounded version)
      - respects `deadline` (a time.monotonic() cutoff): any task not
        started before the deadline is skipped and left to main.py's
        prefilled fallback; tasks already in flight are allowed to finish
        or are cancelled if they'd clearly blow the deadline — document
        which behavior you chose

Edit src/main.py:
  - replace the run_all_noop() call with pipeline.run_all(tasks, settings, deadline)
  - everything else (prefill first, merge, validate, write, catch-all
    return 0) stays as built in Prompt 2 — do not weaken those guarantees

Verification checklist:
[ ] end-to-end run against the 3 published example clips produces valid
    results.json with all requested styles per clip
[ ] forcing ingest.download_video to raise for one task_id still produces
    a full, valid results.json (that task falls back, others succeed)
[ ] forcing grounding.ground_video to raise produces fallback captions for
    only that task, not a pipeline crash
[ ] total wall-clock time for the 3 example clips run concurrently is
    comfortably under settings.max_runtime_seconds
[ ] deadline behavior verified: set an artificially short deadline and
    confirm main.py still exits 0 with a valid (fallback-heavy) output
```

## Prompt 6A — Self-Judge (parallel with 6B)

```
Previously built: src/schemas.py (GroundingFacts, StyledCaption, JudgeScore),
src/config.py (Settings, judge_model field), src/pipeline.py (run_task,
run_all — stable, do not touch in this prompt).

Files to create: src/judge.py only.

Build src/judge.py:
  async def judge_caption(
      facts: GroundingFacts, caption: StyledCaption, settings: Settings
  ) -> JudgeScore
      - MUST call settings.judge_model, which must be configured to be a
        DIFFERENT model than settings.styling_model/settings.grounding_model
        (add a startup-time warning, not a hard failure, if they're equal —
        self-preference bias risk per PROJECT_CONTEXT's judge notes)
      - structured/schema-enforced output mapping to JudgeScore
      - explicit rubric in the prompt: score accuracy and style_match
        independently, explicit instruction to ignore stylistic differences
        when checking core validity (per the plan's judge-reliability notes)
      - on failure, return a neutral JudgeScore(style=caption.style,
        accuracy=1.0, style_match=1.0, notes="judge call failed, no regen")
        so a judge outage never blocks the pipeline or triggers false
        regeneration
  async def judge_and_regenerate(
      facts: GroundingFacts,
      captions: dict[Style, str],
      settings: Settings,
      regenerate_fn,   # Callable[[GroundingFacts, Style, Settings], Awaitable[StyledCaption]]
                        # injected so this file doesn't import styling.py directly
      max_retries: int = 2,
  ) -> dict[Style, str]
      - judges every style in `captions`
      - for any style scoring below an internal threshold (pick and
        document a number, e.g. 0.6 on either axis), calls regenerate_fn
        up to max_retries times, keeping the best-scoring attempt
      - returns a dict with the same keys as `captions`, values possibly
        improved

Verification checklist:
[ ] judge_caption returns scores in [0,1] on both axes for a normal caption
[ ] judge_caption on a caption that clearly doesn't match its stated style
    (e.g. label it "sarcastic" but pass formal text) scores style_match low
[ ] judge_caption on a caption containing a fact not in `facts` scores
    accuracy low
[ ] a forced judge-call failure returns the neutral fallback JudgeScore,
    never raises
[ ] judge_and_regenerate calls regenerate_fn at most max_retries times per
    weak style (verify via call-count spy) and never exceeds the budget
[ ] judge_and_regenerate output always has the same key set as the input
    captions dict
```

## Prompt 6B — Style Rubric Refinement (parallel with 6A)

```
Previously built: src/styling.py (STYLE_RUBRICS dict, style_caption,
style_all — all stable). This prompt only refines the rubric TEXT and adds
exemplars; it does not change any function signature, so it's safe to run
in parallel with judge.py.

Choose ONE:
  Option A (simpler): edit STYLE_RUBRICS directly inside src/styling.py.
  Option B (cleaner parallelism if you're doing this alongside other
  styling.py edits elsewhere): create src/rubrics.py with STYLE_RUBRICS
  there, and have styling.py import it — but note this requires a small
  follow-up edit to styling.py's import line, so it isn't purely parallel
  with 6A unless styling.py is otherwise frozen (it is, per Prompt 5's
  "stable" note above, so Option B is fine here).

Files to create/edit: src/styling.py (Option A) or src/rubrics.py + a
one-line import edit to src/styling.py (Option B).

For each of the 4 styles, expand STYLE_RUBRICS[style] to include:
  - 1 sentence: what tonal markers earn credit
  - 1 sentence: explicit negative constraint (per PROJECT_CONTEXT's
    judge-reliability notes — vague rubrics underperform explicit ones)
  - 2-3 short exemplar captions (not tied to the 3 known example clips —
    keep them generic enough to not bias toward those specific videos)
  - an explicit brevity instruction (target caption length, e.g. "under
    25 words") to guard against verbosity bias

Verification checklist:
[ ] all 4 STYLE_RUBRICS entries still exist under the same exact keys
    (formal, sarcastic, humorous_tech, humorous_non_tech) — nothing renamed
[ ] each entry contains at least one explicit negative constraint
[ ] each entry contains at least 2 exemplars
[ ] re-running style_all against a test clip produces captions noticeably
    tighter/tonally sharper than the Prompt 4C placeholder version (manual
    spot check, side by side)
[ ] no exemplar text leaks verbatim into generated captions (the model is
    using them as style guides, not copying them — spot check)
```

## Prompt 7 — Integrate Judge + Rubrics (sequential, edits pipeline.py)

```
Previously built: src/judge.py (judge_and_regenerate), src/styling.py
(refined STYLE_RUBRICS, style_caption), src/pipeline.py (run_task, run_all
from Prompt 5).

Files to edit: src/pipeline.py only.

Edit run_task in src/pipeline.py:
  - after the existing `styling.style_all(...)` call, add:
      captions = await judge.judge_and_regenerate(
          facts=cleaned_facts,
          captions=captions,
          settings=settings,
          regenerate_fn=lambda f, s, cfg: styling.style_caption(f, s, cfg),
          max_retries=settings.max_self_judge_retries,
      )
  - wrap this call in its own try/except: on failure, fall through with the
    pre-judge captions unchanged (judging is a quality upgrade, never a
    blocker)
  - be mindful of the runtime budget: judge_and_regenerate can roughly
    double or triple the LLM calls for weak styles — factor this into the
    `deadline` check in run_all, or add a lighter per-task time check
    before entering the judge step (skip judging if already close to
    deadline, keep the un-judged captions)

Verification checklist:
[ ] end-to-end run against the 3 example clips still produces valid,
    complete results.json
[ ] forcing judge_and_regenerate to raise leaves pre-judge captions intact
    (no task-level failure)
[ ] total runtime with judging enabled, across the 3 example clips run
    concurrently, is measured and still comfortably under
    settings.max_runtime_seconds — if not, tighten max_self_judge_retries
    or the per-task time check
[ ] spot check: at least one deliberately weak caption in a test run gets
    visibly improved after regeneration (compare before/after)
```

## Prompt 8A — Native Video Input (optional, parallel with 8B)

```
Previously built: src/grounding.py (ground_video — stable, do not modify).
This is an isolated experiment, not wired into pipeline.py yet.

Files to create: src/native_video.py only.

Build src/native_video.py:
  async def ground_video_native(
      video_path: Path, transcript: str | None, ocr_text: list[str], settings: Settings
  ) -> GroundingFacts
      - same output contract as grounding.ground_video, but sends the video
        directly (e.g. a video_url/video content block) to a model that
        supports native video input, instead of pre-extracted frames
      - same GroundingError handling as grounding.py

Verification checklist:
[ ] ground_video_native returns a valid GroundingFacts for the same test
    clips used in Prompt 4A
[ ] side-by-side comparison on a motion-heavy test clip (e.g. sports)
    against grounding.ground_video's frame-based output — document which
    is more accurate before deciding whether to swap it into pipeline.py
[ ] latency/cost of this path measured against the frame-extraction path
    (this determines if it's viable within the runtime budget)
```

## Prompt 8B — Fine-tuning Scaffold (optional, parallel with 8A)

```
Previously built: src/styling.py (style_caption — stable, not modified by
this prompt). This is an isolated experiment directory.

Files to create: finetune/ (new directory, e.g. finetune/prepare_data.py,
finetune/train.py, finetune/README.md) only — does not touch src/ at all.

Build a minimal scaffold to fine-tune a small model purely for the facts→
styled-caption step (not perception), per the plan's Phase 5 note:
  - finetune/prepare_data.py: builds a training set of
    (GroundingFacts, style) → caption pairs, e.g. by sampling style_caption
    outputs from a strong model as teacher labels
  - finetune/train.py: LoRA or DPO training loop against a small base model
  - finetune/README.md: documents time cost, expected payoff, and the
    decision criteria for whether to actually swap this into styling.py

Verification checklist:
[ ] data prep script produces a non-trivial, schema-valid training set
[ ] training run completes without error on a small sample (smoke test,
    not full convergence)
[ ] README clearly states the go/no-go decision inputs (time remaining vs.
    measured quality delta) so this doesn't silently eat the whole budget
```

## Prompt 9 — Hardening Pass (sequential, final)

```
Previously built: full pipeline (src/pipeline.py, src/main.py, all Stage
1/2 modules, judge.py, refined styling.py). Everything is functionally
complete; this prompt only hardens it.

Files to edit: src/main.py, src/pipeline.py. Files to create:
tests/test_hardening.py, a small script (e.g. scripts/check_image.sh) for
final size/build checks.

Tasks:
  - Add concurrency bounds (a semaphore) in pipeline.run_all if not already
    present, sized to avoid provider rate-limit failures under the full
    ~12-clip hidden set
  - Add a simple provider/model fallback: if settings.grounding_model (or
    styling/judge) hits a rate-limit or repeated failure, fall back to a
    secondary configured model rather than exhausting retries against a
    dead endpoint — add SECONDARY_* env vars read in config.load_settings
  - Re-confirm no hardcoded assumption about /input or /output paths beyond
    the documented env-var-with-default pattern from Prompt 2
  - Deliberately break one call path (e.g. monkeypatch grounding.ground_video
    to always raise) in tests/test_hardening.py and assert the full
    container run still exits 0 with valid, complete results.json
  - Check final image size against the 10GB compressed constraint
    (scripts/check_image.sh: docker build, then docker save | gzip | wc -c
    or equivalent, printed clearly)

Verification checklist:
[ ] full run against 3 example clips + a self-assembled variety set
    (sports, food close-up, night/low-light, at minimum) succeeds with
    valid, complete output
[ ] the "deliberately broken" test in tests/test_hardening.py passes
    (exit 0, valid JSON, fallback content present for the broken path)
[ ] measured total runtime for the full concurrent batch stays under
    settings.max_runtime_seconds with margin
[ ] compressed image size confirmed under 10GB, printed by
    scripts/check_image.sh
[ ] no unguaranteed environment variables assumed — every required env var
    either has a default or fails loudly and early in load_settings(),
    never mid-run
[ ] final scan of all prompt text used across grounding/styling/verification/
    judge for language that references the 3 known example clips by name
    or by overly specific description — genericize anything found (per the
    plan's Phase 6 generalization note)
```
