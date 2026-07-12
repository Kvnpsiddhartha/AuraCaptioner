"""Self-judge: score styled captions against `GroundingFacts` on two
independent axes (factual accuracy, style match) and optionally regenerate
weak captions.

Takes `GroundingFacts` and `StyledCaption`/`dict[Style, str]` as input —
does not import styling.py directly. `judge_and_regenerate` instead takes
a `regenerate_fn` callback so this module stays decoupled from however a
caption gets regenerated (pipeline.py wires the real
`styling.style_caption` in at the call site).

Output is REQUIRED to be schema-enforced (tool calling), same as
grounding.py/verification.py/styling.py — no free-text parsing.

Judge reliability notes (per PROJECT_CONTEXT):
  - `settings.judge_model` should be a DIFFERENT model than
    `settings.styling_model`/`settings.grounding_model` to avoid
    self-preference bias. This is only a startup-time WARNING, not a hard
    failure — a hackathon entrant may only have one model available.
  - accuracy and style_match are scored independently; the judge is
    explicitly told to ignore stylistic differences when checking core
    factual validity, so a caption isn't penalized on accuracy just for
    being terse/sarcastic/etc.
  - A judge outage must never block the pipeline or trigger false
    regeneration: any failure to score returns a neutral, maximal
    JudgeScore (never a low one) so nothing is regenerated on a call that
    never actually happened.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from pydantic import BaseModel, Field

from src.config import Settings
from src.schemas import GroundingFacts, JudgeScore, Style, StyledCaption

logger = logging.getLogger(__name__)

JUDGE_API_TIMEOUT_SECONDS = 30.0
JUDGE_MAX_TOKENS = 2048
# One retry on transient failures, mirroring styling.py's budget — judging
# runs once per style already, and judge_and_regenerate can call this
# again per regeneration attempt, so keep individual calls cheap.
JUDGE_MAX_ATTEMPTS = 2
JUDGE_RETRY_BACKOFF_SECONDS = 0.5
# Prompt 9 hardening: after exhausting JUDGE_MAX_ATTEMPTS against the
# primary judge model, try settings.secondary_judge_model (if configured)
# exactly once before falling back to the neutral score. Same rationale
# as grounding.py/styling.py's secondary-model fallback.
SECONDARY_MODEL_TIMEOUT_SECONDS = JUDGE_API_TIMEOUT_SECONDS

# Below this on EITHER axis, a caption is considered weak enough to
# regenerate. Documented threshold (Prompt 6A requirement): 0.6 was chosen
# as a mid-point that flags genuinely weak captions (missing/contradicted
# facts, wrong tone) without triggering regeneration on minor imperfections
# that would just burn the retry budget for no real quality gain.
JUDGE_REGENERATE_THRESHOLD = 0.6

_TOOL_NAME = "record_judge_score"

_SYSTEM_PROMPT = (
    "You are an impartial judge scoring a single video caption against a "
    "factual description of the clip it's meant to describe. Score two "
    "axes INDEPENDENTLY, each from 0.0 to 1.0:\n"
    "  - accuracy: does the caption's content match the facts, with no "
    "invented, contradicted, or missing-core-subject claims? Judge this "
    "ONLY on factual correctness — a terse, sarcastic, or otherwise "
    "stylistically unusual caption can still score high accuracy if "
    "everything it says is true; do not let stylistic differences lower "
    "this score.\n"
    "  - style_match: does the caption's tone/register actually match its "
    "stated style label?\n"
    "Respond by calling the "
    f"`{_TOOL_NAME}` tool exactly once with your scores and a brief note."
)


class _ScoreOutput(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    style_match: float = Field(ge=0, le=1)
    notes: Optional[str] = None


def _score_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Record the accuracy and style_match scores (each 0..1) for "
                "this caption, plus an optional short note."
            ),
            "parameters": _ScoreOutput.model_json_schema(),
        }
    }



def _extract_tool_input(response, tool_name: str) -> dict:
    import json
    message = response.choices[0].message
    if not message.tool_calls:
        raise ValueError("model response did not include any tool calls")
    for tool_call in message.tool_calls:
        if tool_call.type == "function" and tool_call.function.name == tool_name:
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                return json.loads(arguments)
            elif isinstance(arguments, dict):
                return arguments
            raise ValueError(f"tool call for {tool_name!r} had invalid arguments type")
    raise ValueError(f"model response did not include a {tool_name!r} tool call")



def _neutral_score(style: Style) -> JudgeScore:
    """Fallback score used whenever the judge call itself fails. Maximal
    on both axes (never a low score) so a judge outage can never falsely
    trigger regeneration or otherwise block the pipeline."""
    return JudgeScore(
        style=style,
        accuracy=1.0,
        style_match=1.0,
        notes="judge call failed, no regen",
    )


_warned_same_model = False


def _maybe_warn_same_model_as_generator(settings: Settings) -> None:
    """Startup-time WARNING (not a hard failure) if judge_model matches
    styling_model or grounding_model — self-preference bias risk. Only
    logs once per process to avoid spamming per-call."""
    global _warned_same_model
    if _warned_same_model:
        return
    _warned_same_model = True
    if settings.judge_model in (settings.styling_model, settings.grounding_model):
        logger.warning(
            "judge_model=%r matches styling_model/grounding_model — this "
            "risks self-preference bias in judging; configure a distinct "
            "JUDGE_MODEL for reliable self-judging.",
            settings.judge_model,
        )


async def _call_model(
    client,
    facts: GroundingFacts,
    caption: StyledCaption,
    settings: Settings,
    model: str | None = None,
    timeout: float = JUDGE_API_TIMEOUT_SECONDS,
) -> _ScoreOutput:
    response = await client.chat.completions.create(
        model=model or settings.judge_model,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Facts (source of truth):\n{facts.model_dump_json()}\n\n"
                    f"Caption (stated style={caption.style!r}):\n{caption.text}"
                ),
            },
        ],
        tools=[_score_tool_schema()],
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        timeout=timeout,
    )
    tool_input = _extract_tool_input(response, _TOOL_NAME)
    return _ScoreOutput.model_validate(tool_input)



async def judge_caption(
    facts: GroundingFacts, caption: StyledCaption, settings: Settings
) -> JudgeScore:
    """Score `caption` against `facts` on accuracy and style_match.

    Never raises: on any failure (SDK unavailable, API error, malformed
    output) after retries, returns a neutral maximal JudgeScore instead —
    a judge outage must never block the pipeline or trigger false
    regeneration.
    """
    _maybe_warn_same_model_as_generator(settings)

    try:
        from openai import AsyncOpenAI
        import os

        client = AsyncOpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )
    except Exception as exc:
        logger.warning("judge unavailable for style=%s, using neutral score: %s", caption.style, exc)
        return _neutral_score(caption.style)


    last_exc: Exception | None = None
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        try:
            parsed = await _call_model(client, facts, caption, settings)
            return JudgeScore(
                style=caption.style,
                accuracy=parsed.accuracy,
                style_match=parsed.style_match,
                notes=parsed.notes,
            )
        except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
            last_exc = exc
            if attempt < JUDGE_MAX_ATTEMPTS:
                logger.warning(
                    "judge attempt %d/%d failed for style=%s, retrying: %s",
                    attempt, JUDGE_MAX_ATTEMPTS, caption.style, exc,
                )
                await asyncio.sleep(JUDGE_RETRY_BACKOFF_SECONDS)

    logger.warning(
        "judge failed for style=%s after retries, using neutral score: %s",
        caption.style, last_exc,
    )

    if settings.secondary_judge_model:
        logger.warning(
            "judge: primary model=%r exhausted, trying secondary_judge_model=%r "
            "once for style=%s",
            settings.judge_model, settings.secondary_judge_model, caption.style,
        )
        try:
            parsed = await _call_model(
                client, facts, caption, settings,
                model=settings.secondary_judge_model,
                timeout=SECONDARY_MODEL_TIMEOUT_SECONDS,
            )
            return JudgeScore(
                style=caption.style,
                accuracy=parsed.accuracy,
                style_match=parsed.style_match,
                notes=parsed.notes,
            )
        except Exception as exc:
            logger.warning(
                "judge: secondary_judge_model=%r also failed for style=%s: %s",
                settings.secondary_judge_model, caption.style, exc,
            )

    return _neutral_score(caption.style)


RegenerateFn = Callable[[GroundingFacts, Style, Settings], Awaitable[StyledCaption]]


async def _judge_and_maybe_regenerate_one(
    facts: GroundingFacts,
    style: Style,
    text: str,
    settings: Settings,
    regenerate_fn: RegenerateFn,
    max_retries: int,
) -> str:
    """Judge a single (style, text) pair, regenerating up to `max_retries`
    times while it scores below JUDGE_REGENERATE_THRESHOLD on either axis,
    keeping the best-scoring attempt (by axis sum) seen so far."""
    caption = StyledCaption(style=style, text=text)
    best_text = text
    try:
        best_score = await judge_caption(facts, caption, settings)
    except Exception as exc:  # defense-in-depth: judge_caption already never raises
        logger.warning("unexpected exception judging style=%s, keeping as-is: %s", style, exc)
        return text

    attempts = 0
    while (
        (best_score.accuracy < JUDGE_REGENERATE_THRESHOLD or best_score.style_match < JUDGE_REGENERATE_THRESHOLD)
        and attempts < max_retries
    ):
        attempts += 1
        try:
            candidate = await regenerate_fn(facts, style, settings)
        except Exception as exc:
            logger.warning(
                "regenerate_fn failed on attempt %d/%d for style=%s, stopping regen: %s",
                attempts, max_retries, style, exc,
            )
            break

        try:
            candidate_score = await judge_caption(facts, candidate, settings)
        except Exception as exc:
            logger.warning("unexpected exception judging regenerated style=%s: %s", style, exc)
            break

        if (candidate_score.accuracy + candidate_score.style_match) >= (
            best_score.accuracy + best_score.style_match
        ):
            best_text = candidate.text
            best_score = candidate_score

    return best_text


async def judge_and_regenerate(
    facts: GroundingFacts,
    captions: dict[Style, str],
    settings: Settings,
    regenerate_fn: RegenerateFn,
    max_retries: int = 2,
) -> dict[Style, str]:
    """Judge every style in `captions`; regenerate (up to `max_retries`
    times via `regenerate_fn`) any style scoring below
    JUDGE_REGENERATE_THRESHOLD on either axis, keeping the best-scoring
    attempt.

    Judging/regeneration runs concurrently across styles. Always returns a
    dict with exactly the same key set as `captions` — any unexpected
    failure for a given style falls back to that style's original,
    pre-judge text rather than dropping the key or blocking siblings.
    """
    styles = list(captions.keys())
    results = await asyncio.gather(
        *(
            _judge_and_maybe_regenerate_one(
                facts, style, captions[style], settings, regenerate_fn, max_retries
            )
            for style in styles
        ),
        return_exceptions=True,
    )

    out: dict[Style, str] = {}
    for style, result in zip(styles, results):
        if isinstance(result, BaseException):
            logger.warning("unexpected exception in judge_and_regenerate for style=%s: %s", style, result)
            out[style] = captions[style]
        else:
            out[style] = result
    return out
