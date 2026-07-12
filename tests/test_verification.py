"""Tests for src/verification.py — Chain-of-Verification over GroundingFacts."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.schemas import GroundingFacts
from src.verification import verify_facts


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



ORIGINAL_FACTS = GroundingFacts(
    subjects=["a dog", "a unicorn"],  # "a unicorn" is the fabricated claim
    actions=["running"],
    setting="a park",
    mood="playful",
    on_screen_text=[],
    audible_speech=None,
    notable_sounds=["barking"],
)

CLEANED_FACTS_DICT = {
    "subjects": ["a dog"],
    "actions": ["running"],
    "setting": "a park",
    "mood": "playful",
    "on_screen_text": [],
    "audible_speech": None,
    "notable_sounds": ["barking"],
}


def _make_sequenced_client(responses_by_tool: dict[str, dict]):
    """Return a fake client whose chat.completions.create() inspects the forced
    tool_choice on each call and returns the matching canned tool_use
    response — mirrors the real 3-sequential-calls shape of verify_facts."""
    client = mock.Mock()

    async def _create(**kwargs):
        tool_name = kwargs["tool_choice"]["function"]["name"]
        tool_input = responses_by_tool[tool_name]
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall(tool_name, tool_input)]))]
        )

    client.chat.completions.create = mock.AsyncMock(side_effect=_create)
    return client



class TestVerifyFacts:
    def test_drops_fabricated_claim_and_lists_it(self, text_frame):
        responses = {
            "record_verification_questions": {
                "questions": ["Is a unicorn visible in the frames?", "Is a dog visible?"]
            },
            "record_verification_answers": {
                "answers": ["No, no unicorn is visible.", "Yes, a dog is visible."]
            },
            "record_cleaned_facts": {
                "cleaned_facts": CLEANED_FACTS_DICT,
                "dropped_claims": ["a unicorn"],
            },
        }
        client = _make_sequenced_client(responses)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(verify_facts(ORIGINAL_FACTS, [text_frame], _settings()))

        assert "a unicorn" in result.dropped_claims
        assert "a unicorn" not in result.cleaned_facts.subjects
        assert result.original_facts == ORIGINAL_FACTS
        assert result.verification_questions
        assert result.verification_answers

    def test_accurate_facts_pass_through_unchanged_no_false_positive_drops(self, text_frame):
        accurate_facts = GroundingFacts(
            subjects=["a dog"], actions=["running"], setting="a park",
            mood="playful", on_screen_text=[], audible_speech=None,
            notable_sounds=["barking"],
        )
        responses = {
            "record_verification_questions": {"questions": ["Is a dog visible?"]},
            "record_verification_answers": {"answers": ["Yes, a dog is clearly visible."]},
            "record_cleaned_facts": {
                "cleaned_facts": accurate_facts.model_dump(),
                "dropped_claims": [],
            },
        }
        client = _make_sequenced_client(responses)
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(verify_facts(accurate_facts, [text_frame], _settings()))

        assert result.cleaned_facts == accurate_facts
        assert result.dropped_claims == []

    def test_forced_model_failure_falls_back_to_passthrough_never_raises(self, text_frame):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("provider down"))
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(verify_facts(ORIGINAL_FACTS, [text_frame], _settings()))

        assert result.cleaned_facts == ORIGINAL_FACTS
        assert result.dropped_claims == []
        assert result.original_facts == ORIGINAL_FACTS

    def test_failure_in_final_substep_still_falls_back_to_passthrough(self, text_frame):
        """Questions + answers succeed, but the cleaning step fails — the
        whole pass must still fall back to passthrough, not a partial
        result built from only some of the sub-steps."""
        call_count = {"n": 0}

        async def _create(**kwargs):
            tool_name = kwargs["tool_choice"]["function"]["name"]
            call_count["n"] += 1
            if tool_name == "record_verification_questions":
                return _FakeResponse(choices=[
                    _FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall(tool_name, {"questions": ["Is a dog visible?"]})]))
                ])
            if tool_name == "record_verification_answers":
                return _FakeResponse(choices=[
                    _FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall(tool_name, {"answers": ["Yes."]})]))
                ])
            raise RuntimeError("cleaning step exploded")

        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=_create)

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(verify_facts(ORIGINAL_FACTS, [text_frame], _settings()))

        assert result.cleaned_facts == ORIGINAL_FACTS
        assert result.dropped_claims == []

    def test_empty_frames_falls_back_to_passthrough_without_calling_api(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock()
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(verify_facts(ORIGINAL_FACTS, [], _settings()))

        assert result.cleaned_facts == ORIGINAL_FACTS
        client.chat.completions.create.assert_not_awaited()

    def test_answer_count_mismatch_falls_back_to_passthrough(self, text_frame):
        responses = {
            "record_verification_questions": {
                "questions": ["Q1?", "Q2?"]
            },
            "record_verification_answers": {
                "answers": ["only one answer"]  # mismatch: 2 questions, 1 answer
            },
        }
        client = mock.Mock()

        async def _create(**kwargs):
            tool_name = kwargs["tool_choice"]["function"]["name"]
            if tool_name in responses:
                return _FakeResponse(choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall(tool_name, responses[tool_name])]))])
            raise AssertionError("cleaning step should not be reached on mismatch")

        client.chat.completions.create = mock.AsyncMock(side_effect=_create)

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(verify_facts(ORIGINAL_FACTS, [text_frame], _settings()))

        assert result.cleaned_facts == ORIGINAL_FACTS
        assert result.dropped_claims == []
