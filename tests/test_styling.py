"""Tests for src/styling.py — Stage 2 per-style caption generation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.schemas import ALL_STYLES, GroundingFacts
from src.styling import STYLE_RUBRICS, style_all, style_caption


def _settings(**overrides) -> Settings:
    base = dict(
        grounding_model="test-grounding-model",
        styling_model="test-styling-model",
        judge_model="test-judge-model",
        fallback_caption="A short video clip.",
    )
    base.update(overrides)
    return Settings(**base)


FACTS = GroundingFacts(
    subjects=["a dog"], actions=["running"], setting="a park",
    mood="playful", on_screen_text=[], audible_speech=None,
    notable_sounds=["barking"],
)


class _FakeToolCall:
    def __init__(self, name: str, tool_input: dict):
        self.type = "function"
        self.function = mock.Mock()
        self.function.name = name
        import json
        self.function.arguments = json.dumps(tool_input)


class _FakeMessage:
    def __init__(self, tool_calls: list):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, choices: list) -> None:
        self.choices = choices


def _client_returning_text(text_by_style: dict[str, str] | None = None, *, default_text: str | None = None, side_effect=None):
    client = mock.Mock()

    async def _create(**kwargs):
        if side_effect is not None:
            raise side_effect
        # Recover which style this call was for from the prompt text, since
        # style_caption doesn't pass style as an explicit kwarg to the SDK.
        user_content = ""
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        text = default_text
        if text_by_style is not None:
            for style, style_text in text_by_style.items():
                if f"'{style}' style" in user_content:
                    text = style_text
                    break
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("record_caption", {"text": text})]))]
        )

    client.chat.completions.create = mock.AsyncMock(side_effect=_create)
    return client



class TestStyleRubrics:
    def test_all_four_styles_present_with_negative_constraints(self):
        assert set(STYLE_RUBRICS.keys()) == set(ALL_STYLES)
        for style, rubric in STYLE_RUBRICS.items():
            assert "forbid" in rubric.lower(), f"{style} rubric missing an explicit negative constraint"


class TestStyleCaption:
    def test_returns_styled_caption_on_success(self):
        client = _client_returning_text(default_text="A dog sprints through the park.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_caption(FACTS, "formal", _settings()))

        assert result.style == "formal"
        assert result.text == "A dog sprints through the park."

    def test_falls_back_on_repeated_api_failure_never_raises(self):
        client = _client_returning_text(side_effect=RuntimeError("provider down"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_caption(FACTS, "sarcastic", _settings(fallback_caption="fallback!")))

        assert result.style == "sarcastic"
        assert result.text == "fallback!"

    def test_falls_back_on_empty_caption_text(self):
        client = _client_returning_text(default_text="   ")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_caption(FACTS, "humorous_tech", _settings(fallback_caption="fallback!")))

        assert result.text == "fallback!"


class TestStyleAll:
    def test_all_requested_styles_populated_with_non_empty_strings(self):
        client = _client_returning_text(default_text="Something happened in the park.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_all(FACTS, list(ALL_STYLES), _settings()))

        assert set(result.keys()) == set(ALL_STYLES)
        for text in result.values():
            assert text

    def test_captions_differ_across_styles(self):
        by_style = {
            "formal": "The subject proceeds across the designated area.",
            "sarcastic": "Oh look, a dog is running. Riveting.",
            "humorous_tech": "404: Dog.exe found running as expected.",
            "humorous_non_tech": "Zoomies achieved, ten out of ten.",
        }
        client = _client_returning_text(text_by_style=by_style)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_all(FACTS, list(ALL_STYLES), _settings()))

        assert len(set(result.values())) == len(result)  # all distinct
        assert "jargon" not in result["humorous_non_tech"].lower()

    def test_forced_failure_on_one_style_still_returns_all_keys(self):
        async def _create(**kwargs):
            user_content = ""
            for msg in kwargs.get("messages", []):
                if msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            if "'sarcastic' style" in user_content:
                raise RuntimeError("forced failure for sarcastic")
            return _FakeResponse(
                choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("record_caption", {"text": "ok caption"})]))]
            )


        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=_create)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                style_all(FACTS, list(ALL_STYLES), _settings(fallback_caption="fallback!"))
            )

        assert set(result.keys()) == set(ALL_STYLES)
        assert result["sarcastic"] == "fallback!"
        assert result["formal"] == "ok caption"

    def test_caption_lengths_stay_short_across_styles(self):
        client = _client_returning_text(default_text="A short brisk caption about a dog running.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_all(FACTS, list(ALL_STYLES), _settings()))

        for style, text in result.items():
            assert len(text.split()) <= 25, f"{style} caption too long: {text!r}"


class TestSecondaryModelFallback:
    """Prompt 9 hardening: secondary_styling_model is tried once after the
    primary model's retries are exhausted, before falling back to
    fallback_caption."""

    def test_secondary_model_used_after_primary_exhausted(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(
            side_effect=[
                RuntimeError("primary down"),
                RuntimeError("primary down"),
                _FakeResponse(choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("record_caption", {"text": "secondary caption"})]))]),
            ]
        )
        settings = _settings(
            secondary_styling_model="test-secondary-styling-model",
            fallback_caption="fallback!",
        )
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_caption(FACTS, "formal", settings))

        assert result.text == "secondary caption"
        assert client.chat.completions.create.await_count == 3
        last_call_kwargs = client.chat.completions.create.await_args_list[-1].kwargs
        assert last_call_kwargs["model"] == "test-secondary-styling-model"

    def test_falls_back_when_secondary_also_fails(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("everything is down"))
        settings = _settings(
            secondary_styling_model="test-secondary-styling-model",
            fallback_caption="fallback!",
        )
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_caption(FACTS, "formal", settings))

        assert result.text == "fallback!"
        # 2 primary attempts + 1 secondary attempt.
        assert client.chat.completions.create.await_count == 3

    def test_no_secondary_attempt_when_not_configured(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("primary down"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(style_caption(FACTS, "formal", _settings(fallback_caption="fallback!")))

        assert result.text == "fallback!"
        assert client.chat.completions.create.await_count == 2
