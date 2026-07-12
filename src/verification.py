"""Chain-of-Verification (CoVe) pass over `GroundingFacts`.

Takes a `GroundingFacts` object as input as if grounding.py already
produced it — this module does not import grounding.py, so it stays
independently testable and decoupled.

Implements CoVe as three separate schema-enforced sub-steps (never
collapsed into one prompt, per the plan's research basis):
  1. `_generate_questions`  — ask questions that would catch a fabricated
     claim in `facts`.
  2. `_answer_questions`    — answer those questions independently,
     grounded only in the actual frames (not in `facts` itself, so the
     model can't just rubber-stamp its own prior claims).
  3. `_clean_facts`         — given facts + questions + answers, produce
     `cleaned_facts` with any answer-contradicted claims removed and
     listed in `dropped_claims`.

If ANY sub-step fails, `verify_facts` catches it and returns a passthrough
`VerificationResult` (`cleaned_facts == original_facts`, `dropped_claims ==
[]`) rather than raising — verification is a quality upgrade, never a
blocker, so a verification outage must never fail the task.

Model choice: Settings only exposes grounding_model/styling_model/
judge_model (frozen per PROJECT_CONTEXT, no dedicated "verification_model"
field). Sub-step 2 needs to answer questions against actual frames, so all
three sub-steps use settings.grounding_model — the one already established
as vision-capable — rather than judge_model, whose role is caption
scoring, a distinct concern handled in judge.py.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import Settings
from src.schemas import GroundingFacts, VerificationResult

logger = logging.getLogger(__name__)

VERIFICATION_API_TIMEOUT_SECONDS = 45.0
VERIFICATION_MAX_TOKENS = 4096
# Verification only needs enough frames to sanity-check factual claims, not
# the full sampled set — keep this well under grounding's cap to bound
# latency/cost for what is already a 3-call sub-pipeline.
MAX_FRAMES_PER_CALL = 12
MAX_VERIFICATION_QUESTIONS = 6
# One retry per sub-step on transient failures. Kept small since this whole
# pass is already 3 sequential calls and must not eat the task's time
# budget for what is explicitly a "nice to have" accuracy upgrade.
SUBSTEP_MAX_ATTEMPTS = 2
SUBSTEP_RETRY_BACKOFF_SECONDS = 0.5

_QUESTIONS_TOOL_NAME = "record_verification_questions"
_ANSWERS_TOOL_NAME = "record_verification_answers"
_CLEANED_TOOL_NAME = "record_cleaned_facts"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class VerificationError(Exception):
    """Raised between CoVe sub-steps. Never escapes `verify_facts` itself —
    it's always caught there and mapped to a passthrough VerificationResult."""


# --- Local wire-format schemas -------------------------------------------
# These exist only to give each CoVe sub-step a concrete, schema-enforced
# tool-call shape. They are implementation detail of this module, not
# canonical schemas — nothing else imports them.

class _QuestionsOutput(BaseModel):
    questions: list[str] = Field(default_factory=list)


class _AnswersOutput(BaseModel):
    answers: list[str] = Field(default_factory=list)


class _CleanedOutput(BaseModel):
    cleaned_facts: GroundingFacts
    dropped_claims: list[str] = Field(default_factory=list)


def _frame_to_image_block(path: Path) -> dict:
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }



def _select_frames(frames: list[Path], max_frames: int) -> list[Path]:
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    indices = sorted({int(i * step) for i in range(max_frames)})
    return [frames[i] for i in indices]


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



async def _call_with_retry(call_fn):
    """Run `call_fn()` (a zero-arg async callable) up to SUBSTEP_MAX_ATTEMPTS
    times, returning the first success. Re-raises the last exception if
    every attempt fails."""
    last_exc: Exception | None = None
    for attempt in range(1, SUBSTEP_MAX_ATTEMPTS + 1):
        try:
            return await call_fn()
        except Exception as exc:
            last_exc = exc
            if attempt < SUBSTEP_MAX_ATTEMPTS:
                logger.warning(
                    "verification sub-step attempt %d/%d failed, retrying: %s",
                    attempt, SUBSTEP_MAX_ATTEMPTS, exc,
                )
                await asyncio.sleep(SUBSTEP_RETRY_BACKOFF_SECONDS)
    raise last_exc  # type: ignore[misc]


async def _generate_questions(facts: GroundingFacts, settings: Settings, client) -> list[str]:
    """Sub-step 1: ask questions designed to catch a fabricated claim in
    `facts` — grounded only in `facts` itself (the model hasn't seen the
    frames yet at this stage, so questions are pure claim-interrogation)."""

    async def _call():
        response = await client.chat.completions.create(
            model=settings.grounding_model,
            max_tokens=VERIFICATION_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking assistant practicing Chain-of-"
                        "Verification. You will be given a factual description of "
                        "a video clip. Generate up to "
                        f"{MAX_VERIFICATION_QUESTIONS} short, independently-"
                        "checkable yes/no or short-answer questions that would "
                        "expose it if any specific claim in the description were "
                        "fabricated or unsupported (e.g. 'Is there a dog visible "
                        "in the frames?' rather than vague questions). Respond by "
                        f"calling the `{_QUESTIONS_TOOL_NAME}` tool exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Factual description to verify:\n{facts.model_dump_json()}",
                },
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": _QUESTIONS_TOOL_NAME,
                    "description": "Record the verification questions.",
                    "parameters": _QuestionsOutput.model_json_schema(),
                },
            }],
            tool_choice={"type": "function", "function": {"name": _QUESTIONS_TOOL_NAME}},
            timeout=VERIFICATION_API_TIMEOUT_SECONDS,
        )
        tool_input = _extract_tool_input(response, _QUESTIONS_TOOL_NAME)
        parsed = _QuestionsOutput.model_validate(tool_input)
        return parsed.questions[:MAX_VERIFICATION_QUESTIONS]


    return await _call_with_retry(_call)


async def _answer_questions(
    questions: list[str], frames: list[Path], settings: Settings, client
) -> list[str]:
    """Sub-step 2: answer `questions` independently against the actual
    frames — deliberately not shown the original `facts`, so this can't
    just rubber-stamp the claims it's meant to be checking."""
    selected_frames = _select_frames(frames, MAX_FRAMES_PER_CALL)

    async def _call():
        content: list[dict] = [{
            "type": "text",
            "text": (
                "Here are frames sampled from a video clip, followed by a "
                "numbered list of questions. Answer each question in order, "
                "based only on what's visible in these frames — one short "
                "answer per question, same order, same count.\n\nQuestions:\n"
                + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
            ),
        }]
        for path in selected_frames:
            content.append(_frame_to_image_block(path))

        response = await client.chat.completions.create(
            model=settings.grounding_model,
            max_tokens=VERIFICATION_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful visual fact-checker. Answer strictly "
                        "from what is visible in the provided frames; if a "
                        "question can't be answered from the frames, say so rather "
                        "than guessing. Respond by calling the "
                        f"`{_ANSWERS_TOOL_NAME}` tool exactly once, with exactly "
                        f"{len(questions)} answers in the same order as the "
                        "questions."
                    ),
                },
                {"role": "user", "content": content},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": _ANSWERS_TOOL_NAME,
                    "description": "Record the answers, in the same order as the questions.",
                    "parameters": _AnswersOutput.model_json_schema(),
                },
            }],
            tool_choice={"type": "function", "function": {"name": _ANSWERS_TOOL_NAME}},
            timeout=VERIFICATION_API_TIMEOUT_SECONDS,
        )

        tool_input = _extract_tool_input(response, _ANSWERS_TOOL_NAME)
        parsed = _AnswersOutput.model_validate(tool_input)
        if len(parsed.answers) != len(questions):
            raise VerificationError(
                f"expected {len(questions)} answers, got {len(parsed.answers)}"
            )
        return parsed.answers

    return await _call_with_retry(_call)


async def _clean_facts(
    facts: GroundingFacts, questions: list[str], answers: list[str], settings: Settings, client
) -> tuple[GroundingFacts, list[str]]:
    """Sub-step 3: reconcile `facts` against the Q&A pairs, dropping any
    claim the answers don't support."""

    async def _call():
        qa_pairs = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)
        )
        response = await client.chat.completions.create(
            model=settings.grounding_model,
            max_tokens=VERIFICATION_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are reconciling a factual video description against "
                        "independent verification answers. Remove any claim from "
                        "the original description that the answers contradict or "
                        "don't support, and list each removed claim verbatim in "
                        "dropped_claims. Keep everything else unchanged. Respond "
                        f"by calling the `{_CLEANED_TOOL_NAME}` tool exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original description:\n{facts.model_dump_json()}\n\n"
                        f"Verification Q&A:\n{qa_pairs}"
                    ),
                },
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": _CLEANED_TOOL_NAME,
                    "description": "Record the cleaned facts and any dropped claims.",
                    "parameters": _CleanedOutput.model_json_schema(),
                },
            }],
            tool_choice={"type": "function", "function": {"name": _CLEANED_TOOL_NAME}},
            timeout=VERIFICATION_API_TIMEOUT_SECONDS,
        )

        tool_input = _extract_tool_input(response, _CLEANED_TOOL_NAME)
        parsed = _CleanedOutput.model_validate(tool_input)
        return parsed.cleaned_facts, parsed.dropped_claims

    return await _call_with_retry(_call)


def _passthrough(facts: GroundingFacts) -> VerificationResult:
    return VerificationResult(
        original_facts=facts,
        verification_questions=[],
        verification_answers=[],
        cleaned_facts=facts,
        dropped_claims=[],
    )


async def verify_facts(
    facts: GroundingFacts,
    frames: list[Path],
    settings: Settings,
) -> VerificationResult:
    """Run the 3-step Chain-of-Verification pass over `facts`.

    Never raises: any failure in any sub-step (missing SDK, API error,
    schema mismatch) is caught and mapped to a passthrough
    VerificationResult where cleaned_facts == facts and dropped_claims ==
    [] — grounding facts pass through unmodified rather than blocking the
    pipeline.
    """
    try:
        if not frames:
            raise VerificationError("verify_facts called with no frames")

        from openai import AsyncOpenAI
        import os

        client = AsyncOpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )


        questions = await _generate_questions(facts, settings, client)
        if not questions:
            raise VerificationError("no verification questions generated")

        answers = await _answer_questions(questions, frames, settings, client)

        cleaned_facts, dropped_claims = await _clean_facts(
            facts, questions, answers, settings, client
        )

        return VerificationResult(
            original_facts=facts,
            verification_questions=questions,
            verification_answers=answers,
            cleaned_facts=cleaned_facts,
            dropped_claims=dropped_claims,
        )
    except Exception as exc:
        logger.warning("verification pass failed, passing facts through unmodified: %s", exc)
        return _passthrough(facts)
