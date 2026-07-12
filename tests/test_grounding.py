"""Tests for src/grounding.py — Stage 1 VLM grounding into GroundingFacts."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.grounding import GroundingError, ground_video
from src.schemas import GroundingFacts


def _settings(**overrides) -> Settings:
    base = dict(
        grounding_model="test-grounding-model",
        styling_model="test-styling-model",
        judge_model="test-judge-model",
    )
    base.update(overrides)
    return Settings(**base)


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


def _fake_client(*, tool_input=None, tool_name="record_grounding_facts", side_effect=None):
    client = mock.Mock()
    if side_effect is not None:
        client.chat.completions.create = mock.AsyncMock(side_effect=side_effect)
    elif tool_input is None:
        client.chat.completions.create = mock.AsyncMock(
            return_value=_FakeResponse(choices=[_FakeChoice(_FakeMessage(tool_calls=[]))])
        )
    else:
        client.chat.completions.create = mock.AsyncMock(
            return_value=_FakeResponse(
                choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall(tool_name, tool_input)]))]
            )
        )
    return client



VALID_FACTS_INPUT = {
    "subjects": ["a dog"],
    "actions": ["running across a field"],
    "setting": "an outdoor park",
    "mood": "playful",
    "on_screen_text": [],
    "audible_speech": None,
    "notable_sounds": ["barking"],
}


class TestGroundVideo:
    def test_returns_valid_grounding_facts_for_a_known_clip(self, text_frame):
        client = _fake_client(tool_input=VALID_FACTS_INPUT)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(ground_video([text_frame], None, [], _settings()))

        assert isinstance(result, GroundingFacts)
        assert result.subjects == ["a dog"]
        assert result.actions
        assert result.setting

    def test_works_with_no_transcript_and_no_ocr_text(self, text_frame):
        client = _fake_client(tool_input=VALID_FACTS_INPUT)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(ground_video([text_frame], transcript=None, ocr_text=[], settings=_settings()))

        assert isinstance(result, GroundingFacts)
        # The call must still have happened (visual-only grounding), not skipped.
        client.chat.completions.create.assert_awaited_once()

    def test_raises_grounding_error_on_missing_tool_call(self, text_frame):
        client = _fake_client(tool_input=None)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([text_frame], None, [], _settings()))

    def test_raises_grounding_error_on_malformed_schema(self, text_frame):
        # Missing/wrong-typed fields should fail GroundingFacts validation.
        client = _fake_client(tool_input={"subjects": "not-a-list"})
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([text_frame], None, [], _settings()))

    def test_raises_grounding_error_on_api_failure_after_retries(self, text_frame):
        client = _fake_client(side_effect=RuntimeError("connection reset"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([text_frame], None, [], _settings()))
        # One initial attempt + one retry, per GROUNDING_MAX_ATTEMPTS.
        assert client.chat.completions.create.await_count == 2

    def test_raises_grounding_error_on_empty_frames_without_calling_api(self):
        client = _fake_client(tool_input=VALID_FACTS_INPUT)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([], None, [], _settings()))
        client.chat.completions.create.assert_not_awaited()

    def test_does_not_leak_unhandled_exception_types(self, text_frame):
        """Whatever goes wrong internally, callers only ever see GroundingError."""
        client = _fake_client(side_effect=ValueError("boom"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([text_frame], None, [], _settings()))


class TestSecondaryModelFallback:
    """Prompt 9 hardening: secondary_grounding_model is tried once after
    the primary model's retries are exhausted."""

    def test_secondary_model_used_after_primary_exhausted(self, text_frame):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(
            side_effect=[
                RuntimeError("primary down"),
                RuntimeError("primary down"),
                _FakeResponse(choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("record_grounding_facts", VALID_FACTS_INPUT)]))]),
            ]
        )
        settings = _settings(secondary_grounding_model="test-secondary-grounding-model")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(ground_video([text_frame], None, [], settings))

        assert isinstance(result, GroundingFacts)
        assert client.chat.completions.create.await_count == 3
        last_call_kwargs = client.chat.completions.create.await_args_list[-1].kwargs
        assert last_call_kwargs["model"] == "test-secondary-grounding-model"

    def test_raises_grounding_error_when_secondary_also_fails(self, text_frame):
        client = _fake_client(side_effect=RuntimeError("everything is down"))
        settings = _settings(secondary_grounding_model="test-secondary-grounding-model")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([text_frame], None, [], settings))
        # 2 primary attempts + 1 secondary attempt.
        assert client.chat.completions.create.await_count == 3

    def test_no_secondary_attempt_when_not_configured(self, text_frame):
        client = _fake_client(side_effect=RuntimeError("primary down"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video([text_frame], None, [], _settings()))
        # No secondary_grounding_model configured -> only primary attempts.
        assert client.chat.completions.create.await_count == 2
