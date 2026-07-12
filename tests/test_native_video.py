"""Tests for src/native_video.py — the isolated native-video grounding
experiment from Prompt 8A."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.grounding import GroundingError
from src.native_video import NATIVE_VIDEO_MAX_BYTES, ground_video_native
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


class TestGroundVideoNative:
    def test_returns_valid_grounding_facts_for_a_real_video_file(self, multi_cut_video):
        client = _fake_client(tool_input=VALID_FACTS_INPUT)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(ground_video_native(multi_cut_video, None, [], _settings()))

        assert isinstance(result, GroundingFacts)
        assert result.subjects == ["a dog"]
        assert result.setting

    def test_raises_grounding_error_on_missing_video_path(self):
        with pytest.raises(GroundingError):
            asyncio.run(
                ground_video_native(Path("/tmp/does_not_exist_xyz.mp4"), None, [], _settings())
            )

    def test_raises_grounding_error_when_video_exceeds_size_cap(self, tmp_path):
        oversized = tmp_path / "too_big.mp4"
        oversized.write_bytes(b"0" * (NATIVE_VIDEO_MAX_BYTES + 1))
        with pytest.raises(GroundingError):
            asyncio.run(ground_video_native(oversized, None, [], _settings()))

    def test_oversized_payload_is_not_retried(self, tmp_path, monkeypatch):
        """Oversized-payload failures aren't transient, so they should not
        burn the retry budget by calling the model at all."""
        oversized = tmp_path / "too_big.mp4"
        oversized.write_bytes(b"0" * (NATIVE_VIDEO_MAX_BYTES + 1))

        client = _fake_client(tool_input=VALID_FACTS_INPUT)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video_native(oversized, None, [], _settings()))
        client.chat.completions.create.assert_not_called()

    def test_raises_grounding_error_on_malformed_tool_call(self, multi_cut_video):
        client = _fake_client(tool_input=None)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video_native(multi_cut_video, None, [], _settings()))

    def test_raises_grounding_error_when_sdk_unavailable(self, multi_cut_video, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("no openai SDK installed")

            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(GroundingError):
            asyncio.run(ground_video_native(multi_cut_video, None, [], _settings()))

    def test_retries_once_on_transient_failure_then_succeeds(self, multi_cut_video):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(
            side_effect=[
                TimeoutError("transient timeout"),
                _FakeResponse(choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("record_grounding_facts", VALID_FACTS_INPUT)]))]),
            ]
        )
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(ground_video_native(multi_cut_video, "hello", ["SALE"], _settings()))

        assert isinstance(result, GroundingFacts)
        assert client.chat.completions.create.await_count == 2

    def test_same_failure_contract_as_grounding_error_type(self, multi_cut_video):
        """A future pipeline.py integration should be able to catch this
        with the exact same except clause used for grounding.ground_video."""
        client = _fake_client(side_effect=RuntimeError("boom"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            with pytest.raises(GroundingError):
                asyncio.run(ground_video_native(multi_cut_video, None, [], _settings()))
