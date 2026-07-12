"""Prompt 8A (optional, experimental): ground a clip by sending the video
file itself to a model with native video-understanding support, instead of
grounding.py's pre-extracted-frames path.

This module is deliberately ISOLATED: it is not imported by pipeline.py and
does not change any existing behavior. It exists so the two grounding
strategies (frame-based vs native-video) can be measured side by side
before deciding whether to swap this into the real pipeline (per Prompt
8A's verification checklist). It reuses `grounding.GroundingError` so a
future integration into pipeline.py can catch exactly the same exception
type pipeline.py already catches for `grounding.ground_video`.

Same contract as grounding.ground_video:
  - output is REQUIRED to be schema-enforced (tool calling), never
    free-text parsing
  - never returns a partially-filled/guessed GroundingFacts — any failure
    (missing input, SDK/API error, timeout, oversized payload, malformed/
    missing tool call, schema validation failure) raises GroundingError
    with task context, exactly as grounding.ground_video does
"""
from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.config import Settings
from src.grounding import GroundingError
from src.schemas import GroundingFacts

logger = logging.getLogger(__name__)

NATIVE_VIDEO_API_TIMEOUT_SECONDS = 90.0
NATIVE_VIDEO_MAX_TOKENS = 4096
# One retry on transient failures, mirroring grounding.py's budget.
NATIVE_VIDEO_MAX_ATTEMPTS = 2
NATIVE_VIDEO_RETRY_BACKOFF_SECONDS = 1.0

# Defensive upper bound on the raw video payload sent as a single inline
# base64 block. This is an experimental path with no frame-extraction
# pre-processing to shrink the payload, so a generous but explicit cap
# matters more here than in grounding.py (which only ever sends small
# still frames). ingest.downscale already keeps clips small in the real
# pipeline, but this module intentionally does not import ingest.py (see
# module docstring), so it re-checks size itself rather than trusting the
# caller.
NATIVE_VIDEO_MAX_BYTES = 20 * 1024 * 1024  # 20MB

_TOOL_NAME = "record_grounding_facts"

_MEDIA_TYPE_FALLBACK = "video/mp4"

_SYSTEM_PROMPT = (
    "You are a precise video-grounding assistant. You will be shown a "
    "short video clip directly (not individual frames), plus optionally a "
    "speech transcript and any on-screen text already detected by OCR. "
    "Your job is ONLY to record what is actually visible/audible in the "
    "video — do not infer, assume, or invent anything not directly "
    "supported by the video, transcript, or OCR text provided. If "
    "something is unclear or ambiguous, describe it conservatively rather "
    "than guessing. You must respond by calling the "
    f"`{_TOOL_NAME}` tool exactly once with your findings; do not respond "
    "with plain text."
)


def _facts_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Record the style-agnostic factual description of the video "
                "clip shown, grounded only in what is actually visible/audible."
            ),
            "parameters": GroundingFacts.model_json_schema(),
        }
    }



def _video_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("video/"):
        return guessed
    return _MEDIA_TYPE_FALLBACK


def _video_to_content_block(path: Path) -> dict:
    """Build a native video content block from a local file.

    Uses the same base64-inline shape as grounding.py's image blocks, just
    with `type: "video_url"` instead of `type: "image"` — the exact wire
    format a given provider expects for native video input may differ, so
    this is intentionally isolated behind this one function: swapping
    providers only means changing this helper.
    """
    size = path.stat().st_size
    if size > NATIVE_VIDEO_MAX_BYTES:
        raise GroundingError(
            f"native video payload for {path} is {size} bytes, exceeding "
            f"the {NATIVE_VIDEO_MAX_BYTES}-byte cap for this experimental "
            "inline path — downscale further or skip native grounding for "
            "this clip"
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{_video_media_type(path)};base64,{data}"},
    }



def _build_user_content(
    video_path: Path, transcript: Optional[str], ocr_text: list[str]
) -> list[dict]:
    context_lines = []
    if transcript:
        context_lines.append(f"Speech transcript (may be empty/partial): {transcript!r}")
    else:
        context_lines.append("Speech transcript: none detected.")
    if ocr_text:
        context_lines.append(f"On-screen text already detected by OCR: {ocr_text!r}")
    else:
        context_lines.append("On-screen text: none detected.")

    return [
        {
            "type": "text",
            "text": (
                "Here is a short video clip, shown directly (not as "
                "extracted frames).\n" + "\n".join(context_lines)
            ),
        },
        _video_to_content_block(video_path),
    ]


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



async def _call_model(
    client, video_path: Path, transcript: Optional[str], ocr_text: list[str], settings: Settings
) -> dict:
    response = await client.chat.completions.create(
        model=settings.grounding_model,
        max_tokens=NATIVE_VIDEO_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_content(video_path, transcript, ocr_text)},
        ],
        tools=[_facts_tool_schema()],
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        timeout=NATIVE_VIDEO_API_TIMEOUT_SECONDS,
    )
    return _extract_tool_input(response, _TOOL_NAME)



async def ground_video_native(
    video_path: Path,
    transcript: Optional[str],
    ocr_text: list[str],
    settings: Settings,
) -> GroundingFacts:
    """Ground `video_path` (+ optional transcript/ocr_text) into a
    `GroundingFacts` instance by sending the video directly to a model
    with native video support, instead of grounding.ground_video's
    pre-extracted-frames approach.

    Raises GroundingError (never a raw SDK/pydantic exception) if the
    video file is missing, oversized for this experimental inline-payload
    path, the SDK is unavailable, the API call fails on every attempt, or
    the model's tool call doesn't validate against GroundingFacts — the
    exact same failure contract as grounding.ground_video, so a future
    caller can treat the two functions interchangeably.
    """
    if not video_path.exists():
        raise GroundingError(f"ground_video_native called with missing video_path={video_path}")

    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover - exercised via SDK-missing tests
        raise GroundingError(f"openai SDK not available: {exc}") from exc

    try:
        import os
        client = AsyncOpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )
    except Exception as exc:
        raise GroundingError(f"failed to construct Fireworks client: {exc}") from exc


    last_exc: Exception | None = None
    for attempt in range(1, NATIVE_VIDEO_MAX_ATTEMPTS + 1):
        try:
            tool_input = await _call_model(client, video_path, transcript, ocr_text, settings)
            return GroundingFacts.model_validate(tool_input)
        except GroundingError:
            # Oversized-payload is not transient — retrying won't shrink
            # the file — so surface it immediately rather than burning
            # the retry budget.
            raise
        except ValidationError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc

        if attempt < NATIVE_VIDEO_MAX_ATTEMPTS:
            logger.warning(
                "native_video grounding attempt %d/%d failed, retrying: %s",
                attempt, NATIVE_VIDEO_MAX_ATTEMPTS, last_exc,
            )
            await asyncio.sleep(NATIVE_VIDEO_RETRY_BACKOFF_SECONDS)

    raise GroundingError(
        f"native video grounding failed after {NATIVE_VIDEO_MAX_ATTEMPTS} attempt(s) "
        f"using model={settings.grounding_model!r}: {last_exc}"
    ) from last_exc
