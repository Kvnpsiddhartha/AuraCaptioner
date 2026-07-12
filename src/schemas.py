"""Canonical schemas for the video captioning agent.

These models are treated as frozen once written here — downstream modules
(ingest, sampling, grounding, verification, styling, judge, pipeline)
import from this file rather than redefining shapes locally.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Style = Literal["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

ALL_STYLES: tuple[Style, ...] = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)


class Task(BaseModel):
    """A single input task from /input/tasks.json."""

    task_id: str
    video_url: str
    styles: list[Style]


class GroundingFacts(BaseModel):
    """Style-agnostic factual description of a video (Stage 1 output)."""

    subjects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    setting: str = ""
    mood: str = ""
    on_screen_text: list[str] = Field(default_factory=list)
    audible_speech: Optional[str] = None
    notable_sounds: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Output of the Chain-of-Verification pass over GroundingFacts."""

    original_facts: GroundingFacts
    verification_questions: list[str] = Field(default_factory=list)
    verification_answers: list[str] = Field(default_factory=list)
    cleaned_facts: GroundingFacts
    dropped_claims: list[str] = Field(default_factory=list)


class StyledCaption(BaseModel):
    """A single style's caption for a task."""

    style: Style
    text: str


class JudgeScore(BaseModel):
    """Self-judge score for one styled caption, mirroring the real judge's axes."""

    style: Style
    accuracy: float = Field(ge=0, le=1)
    style_match: float = Field(ge=0, le=1)
    notes: Optional[str] = None


class TaskResult(BaseModel):
    """A single output entry for /output/results.json."""

    task_id: str
    captions: dict[Style, str]
