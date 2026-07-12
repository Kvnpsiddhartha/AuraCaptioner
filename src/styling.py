"""Stage 2 styling: turn a style-agnostic `GroundingFacts` into a short,
on-style caption for each requested `Style`.

Takes a `GroundingFacts` object as input — does not import grounding.py or
verification.py, so this stays independently testable and decoupled from
however the facts were produced.

Each style's caption is a single schema-enforced call, grounded ONLY in
`facts` (never re-derive facts from the style prompt — style should change
tone, not content). Styling failures are per-style: a failure for one
style must never block or affect the others, so `style_caption` always
returns a `StyledCaption` (falling back to `settings.fallback_caption`
rather than raising) and `style_all` runs every requested style
concurrently and always returns every requested key.
"""
from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from src.config import Settings
from src.schemas import GroundingFacts, Style, StyledCaption

logger = logging.getLogger(__name__)

STYLING_API_TIMEOUT_SECONDS = 30.0
STYLING_MAX_TOKENS = 2048
# One retry on transient failures. Kept short since style_all fans this out
# concurrently across (up to 4) styles per task — a slow retry loop here
# multiplies straight into the task's wall-clock time.
STYLING_MAX_ATTEMPTS = 2
STYLING_RETRY_BACKOFF_SECONDS = 0.5
# Prompt 9 hardening: after exhausting STYLING_MAX_ATTEMPTS against the
# primary model, try settings.secondary_styling_model (if configured)
# exactly once before falling back to settings.fallback_caption. Same
# rationale as grounding.py's secondary-model fallback: cheap insurance
# against a provider-wide outage/rate-limit that same-model retries can't
# fix.
SECONDARY_MODEL_TIMEOUT_SECONDS = STYLING_API_TIMEOUT_SECONDS

_TOOL_NAME = "record_caption"

# Refined per Prompt 6B: each rubric now states (a) the tonal markers that
# earn credit, (b) an explicit negative constraint (vague rubrics
# underperform explicit ones for LLM judging/generation reliability), (c)
# 2-3 generic exemplars (deliberately NOT tied to any specific known clip,
# so the model can't overfit to one scene), and (d) an explicit brevity
# instruction to counter length/verbosity bias. The dict SHAPE (same 4
# keys) is unchanged from the Prompt 4C placeholder — other prompts
# (6A/6B, pipeline.py) assume these exact keys.
STYLE_RUBRICS: dict[Style, str] = {
    "formal": (
        "Rewards precise, neutral, structured phrasing suitable for a news "
        "caption or archival description — third person, complete "
        "sentence, objective word choice. Forbid slang, contractions, "
        "humor of any kind, exclamation marks, and first/second person "
        "address. Target length: under 20 words, one sentence. "
        "Exemplars: 'A cyclist crosses an intersection during light "
        "midday traffic.' / 'Two individuals prepare a meal in a "
        "residential kitchen.' / 'A crowd gathers outdoors as a speaker "
        "addresses them from a small stage.'"
    ),
    "sarcastic": (
        "Rewards dry understatement, deadpan irony, and a wry reframing of "
        "the plainly obvious. Forbid exclamation marks, enthusiastic or "
        "gushing adjectives ('amazing', 'incredible'), and any sincerity — "
        "if a line could be read as genuine praise, it isn't sarcastic "
        "enough. Target length: under 20 words, one line. Exemplars: "
        "'Riveting stuff: a man walks to a car, gets in, drives away.' / "
        "'Truly the pinnacle of human achievement: someone made toast.' / "
        "'Groundbreaking footage of a dog doing exactly what dogs do.'"
    ),
    "humorous_tech": (
        "Rewards tech-culture references, developer/engineering wordplay, "
        "and jokes that explicitly assume a technical audience (bugs, "
        "deploys, latency, APIs, version numbers, framework in-jokes). "
        "Forbid captions that read as generic humor with no tech angle — "
        "if the joke works equally well as humorous_non_tech, rewrite it "
        "with an actual technical hook. Also forbid jargon dumped without "
        "a joke attached to it. Target length: under 22 words. Exemplars: "
        "'404: chill not found — this cat refuses to load.' / 'Merge "
        "conflict IRL: two dogs, one couch, zero consensus.' / 'Deploying "
        "to production on a Friday energy, but it's just someone parking a "
        "car.'"
    ),
    "humorous_non_tech": (
        "Rewards accessible, everyday humor anyone would get on first "
        "read — relatable observations, gentle exaggeration, playful "
        "commentary. Forbid jargon, technical terminology, or any "
        "reference to code/software/engineering entirely — if a "
        "non-technical friend would need something explained, cut it. "
        "Target length: under 20 words. Exemplars: 'This dog has never "
        "once considered walking in a straight line.' / 'Breaking: local "
        "man successfully finds parking spot, crowd unimpressed.' / "
        "'She's not late, she's simply on a different clock than the rest "
        "of us.'"
    ),
}


class _CaptionOutput(BaseModel):
    text: str


def _caption_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Record the styled caption for this clip.",
            "parameters": _CaptionOutput.model_json_schema(),
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



def _build_prompt(facts: GroundingFacts, style: Style) -> str:
    return (
        f"Facts about the clip (the ONLY source of truth — do not add "
        f"anything not present here):\n{facts.model_dump_json()}\n\n"
        f"Write ONE short caption in the '{style}' style. Keep it brief "
        "(well under 25 words) — a single punchy sentence, not a "
        "paragraph. Respond by calling the "
        f"`{_TOOL_NAME}` tool exactly once with the caption text."
    )


async def _call_model(
    client,
    facts: GroundingFacts,
    style: Style,
    settings: Settings,
    model: str | None = None,
    timeout: float = STYLING_API_TIMEOUT_SECONDS,
) -> str:
    response = await client.chat.completions.create(
        model=model or settings.styling_model,
        max_tokens=STYLING_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a caption writer. You only ever have access to the "
                    "structured facts you're given — never invent details. "
                    f"Style rubric for '{style}': {STYLE_RUBRICS[style]}"
                ),
            },
            {"role": "user", "content": _build_prompt(facts, style)},
        ],
        tools=[_caption_tool_schema()],
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
        timeout=timeout,
    )
    tool_input = _extract_tool_input(response, _TOOL_NAME)
    parsed = _CaptionOutput.model_validate(tool_input)
    text = parsed.text.strip()
    if not text:
        raise ValueError("model returned an empty caption")
    return text



async def style_caption(facts: GroundingFacts, style: Style, settings: Settings) -> StyledCaption:
    """Produce a single styled caption for `style`, grounded only in
    `facts`. Never raises: on any failure (SDK unavailable, API error,
    empty/invalid output) after all retries, returns
    StyledCaption(style=style, text=settings.fallback_caption) instead —
    a styling failure for one style must never block sibling styles."""
    try:
        from openai import AsyncOpenAI
        import os

        client = AsyncOpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )
    except Exception as exc:
        logger.warning("styling unavailable for style=%s, using fallback: %s", style, exc)
        return StyledCaption(style=style, text=settings.fallback_caption)


    last_exc: Exception | None = None
    for attempt in range(1, STYLING_MAX_ATTEMPTS + 1):
        try:
            text = await _call_model(client, facts, style, settings)
            return StyledCaption(style=style, text=text)
        except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
            last_exc = exc
            if attempt < STYLING_MAX_ATTEMPTS:
                logger.warning(
                    "styling attempt %d/%d failed for style=%s, retrying: %s",
                    attempt, STYLING_MAX_ATTEMPTS, style, exc,
                )
                await asyncio.sleep(STYLING_RETRY_BACKOFF_SECONDS)

    if settings.secondary_styling_model:
        logger.warning(
            "styling: primary model=%r exhausted, trying "
            "secondary_styling_model=%r once for style=%s",
            settings.styling_model, settings.secondary_styling_model, style,
        )
        try:
            text = await _call_model(
                client, facts, style, settings,
                model=settings.secondary_styling_model,
                timeout=SECONDARY_MODEL_TIMEOUT_SECONDS,
            )
            return StyledCaption(style=style, text=text)
        except Exception as exc:
            logger.warning(
                "styling: secondary_styling_model=%r also failed for style=%s: %s",
                settings.secondary_styling_model, style, exc,
            )

    logger.warning("styling failed for style=%s after retries, using fallback: %s", style, last_exc)
    return StyledCaption(style=style, text=settings.fallback_caption)


async def style_all(
    facts: GroundingFacts, styles: list[Style], settings: Settings
) -> dict[Style, str]:
    """Run style_caption concurrently for every style in `styles`.

    Guaranteed to return every requested style as a key (style_caption's
    own fallback behavior means no individual failure can drop a key, and
    this function adds a defense-in-depth catch-all so even an unexpected
    exception escaping style_caption can't drop a key either).
    """
    results = await asyncio.gather(
        *(style_caption(facts, style, settings) for style in styles),
        return_exceptions=True,
    )

    captions: dict[Style, str] = {}
    for style, result in zip(styles, results):
        if isinstance(result, BaseException):
            logger.warning("unexpected exception styling style=%s, using fallback: %s", style, result)
            captions[style] = settings.fallback_caption
        else:
            captions[style] = result.text
    return captions
