"""Tests for src/judge.py — self-judge scoring and judge-and-regenerate."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.schemas import GroundingFacts, StyledCaption
from src.judge import judge_and_regenerate, judge_caption, JUDGE_REGENERATE_THRESHOLD


def _settings(**overrides) -> Settings:
    base = dict(
        grounding_model="test-grounding-model",
        styling_model="test-styling-model",
        judge_model="test-judge-model",
        fallback_caption="A short video clip.",
        max_self_judge_retries=2,
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


def _client_with_scores(scores: list[dict] | None = None, *, fixed: dict | None = None, side_effect=None):
    """Fake client. `scores` pops one score dict per call
    (in order); `fixed` returns the same score every call."""
    call_log: list[dict] = []
    scores_iter = iter(scores) if scores is not None else None

    client = mock.Mock()

    async def _create(**kwargs):
        call_log.append(kwargs)
        if side_effect is not None:
            raise side_effect
        if scores_iter is not None:
            score = next(scores_iter)
        else:
            score = fixed
        return _FakeResponse(
            choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall("record_judge_score", score)]))]
        )

    client.chat.completions.create = mock.AsyncMock(side_effect=_create)
    client._call_log = call_log
    return client



class TestJudgeCaption:
    def test_returns_scores_in_range(self):
        client = _client_with_scores(fixed={"accuracy": 0.9, "style_match": 0.8, "notes": "good"})
        caption = StyledCaption(style="formal", text="A dog runs in a park.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            score = asyncio.run(judge_caption(FACTS, caption, _settings()))

        assert score.style == "formal"
        assert 0.0 <= score.accuracy <= 1.0
        assert 0.0 <= score.style_match <= 1.0

    def test_mismatched_style_scores_style_match_low(self):
        # Simulates a judge correctly identifying that formal text was
        # mislabeled as sarcastic.
        client = _client_with_scores(fixed={"accuracy": 0.9, "style_match": 0.1, "notes": "not sarcastic at all"})
        caption = StyledCaption(style="sarcastic", text="A dog proceeds across the park in a northerly direction.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            score = asyncio.run(judge_caption(FACTS, caption, _settings()))

        assert score.style_match < 0.6

    def test_forced_failure_returns_neutral_fallback_never_raises(self):
        client = _client_with_scores(side_effect=RuntimeError("judge provider down"))
        caption = StyledCaption(style="formal", text="A dog runs in a park.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            score = asyncio.run(judge_caption(FACTS, caption, _settings()))

        assert score.style == "formal"
        assert score.accuracy == 1.0
        assert score.style_match == 1.0
        assert score.notes is not None


class TestSecondaryModelFallback:
    """Prompt 9 hardening: secondary_judge_model is tried once after the
    primary judge model's retries are exhausted, before falling back to
    the neutral score."""

    def test_secondary_model_used_after_primary_exhausted(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(
            side_effect=[
                RuntimeError("primary judge down"),
                RuntimeError("primary judge down"),
                _FakeResponse(
                    choices=[_FakeChoice(_FakeMessage(tool_calls=[_FakeToolCall(
                        "record_judge_score", {"accuracy": 0.95, "style_match": 0.9, "notes": "from secondary"},
                    )]))]
                ),
            ]
        )
        caption = StyledCaption(style="formal", text="A dog runs in a park.")
        settings = _settings(secondary_judge_model="test-secondary-judge-model")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            score = asyncio.run(judge_caption(FACTS, caption, settings))

        assert score.accuracy == 0.95
        assert score.style_match == 0.9
        assert client.chat.completions.create.await_count == 3
        last_call_kwargs = client.chat.completions.create.await_args_list[-1].kwargs
        assert last_call_kwargs["model"] == "test-secondary-judge-model"

    def test_falls_back_to_neutral_when_secondary_also_fails(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("everything down"))
        caption = StyledCaption(style="formal", text="A dog runs in a park.")
        settings = _settings(secondary_judge_model="test-secondary-judge-model")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            score = asyncio.run(judge_caption(FACTS, caption, settings))

        assert score.accuracy == 1.0
        assert score.style_match == 1.0
        assert client.chat.completions.create.await_count == 3

    def test_no_secondary_attempt_when_not_configured(self):
        client = mock.Mock()
        client.chat.completions.create = mock.AsyncMock(side_effect=RuntimeError("primary judge down"))
        caption = StyledCaption(style="formal", text="A dog runs in a park.")
        with mock.patch("openai.AsyncOpenAI", return_value=client):
            score = asyncio.run(judge_caption(FACTS, caption, _settings()))

        assert score.accuracy == 1.0
        assert client.chat.completions.create.await_count == 2


class TestJudgeAndRegenerate:
    def test_regenerate_fn_called_at_most_max_retries_per_weak_style(self):
        # judge always returns a weak score, so every retry budget should
        # be fully exhausted, never exceeded.
        client = _client_with_scores(fixed={"accuracy": 0.2, "style_match": 0.2, "notes": "weak"})
        call_count = {"n": 0}

        async def regenerate_fn(facts, style, settings):
            call_count["n"] += 1
            return StyledCaption(style=style, text=f"regenerated attempt {call_count['n']}")

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                judge_and_regenerate(
                    facts=FACTS,
                    captions={"formal": "original weak caption"},
                    settings=_settings(),
                    regenerate_fn=regenerate_fn,
                    max_retries=2,
                )
            )

        assert call_count["n"] <= 2
        assert set(result.keys()) == {"formal"}

    def test_strong_caption_is_not_regenerated(self):
        client = _client_with_scores(fixed={"accuracy": 0.95, "style_match": 0.95, "notes": "great"})
        regenerate_calls = {"n": 0}

        async def regenerate_fn(facts, style, settings):
            regenerate_calls["n"] += 1
            return StyledCaption(style=style, text="should not be used")

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                judge_and_regenerate(
                    facts=FACTS,
                    captions={"formal": "already great caption"},
                    settings=_settings(),
                    regenerate_fn=regenerate_fn,
                    max_retries=2,
                )
            )

        assert regenerate_calls["n"] == 0
        assert result["formal"] == "already great caption"

    def test_keeps_best_scoring_attempt(self):
        # Sequence of scores as judge_caption is called: original (weak),
        # regen #1 (worse), regen #2 (better than original) -> final
        # result should be regen #2's text.
        score_sequence = [
            {"accuracy": 0.4, "style_match": 0.4, "notes": "original"},
            {"accuracy": 0.2, "style_match": 0.2, "notes": "regen1 worse"},
            {"accuracy": 0.9, "style_match": 0.9, "notes": "regen2 best"},
        ]
        client = _client_with_scores(scores=score_sequence)

        regen_texts = iter(["regen one text", "regen two text"])

        async def regenerate_fn(facts, style, settings):
            return StyledCaption(style=style, text=next(regen_texts))

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                judge_and_regenerate(
                    facts=FACTS,
                    captions={"formal": "original text"},
                    settings=_settings(),
                    regenerate_fn=regenerate_fn,
                    max_retries=2,
                )
            )

        assert result["formal"] == "regen two text"

    def test_output_keys_match_input_keys(self):
        client = _client_with_scores(fixed={"accuracy": 0.95, "style_match": 0.95, "notes": "fine"})
        captions = {
            "formal": "a",
            "sarcastic": "b",
            "humorous_tech": "c",
            "humorous_non_tech": "d",
        }

        async def regenerate_fn(facts, style, settings):
            return StyledCaption(style=style, text="regenerated")

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                judge_and_regenerate(
                    facts=FACTS,
                    captions=captions,
                    settings=_settings(),
                    regenerate_fn=regenerate_fn,
                    max_retries=2,
                )
            )

        assert set(result.keys()) == set(captions.keys())

    def test_regenerate_fn_failure_keeps_previous_best_not_raise(self):
        client = _client_with_scores(fixed={"accuracy": 0.1, "style_match": 0.1, "notes": "weak"})

        async def regenerate_fn(facts, style, settings):
            raise RuntimeError("styling provider down")

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                judge_and_regenerate(
                    facts=FACTS,
                    captions={"formal": "original text"},
                    settings=_settings(),
                    regenerate_fn=regenerate_fn,
                    max_retries=2,
                )
            )

        assert result["formal"] == "original text"

    def test_judge_outage_never_triggers_regeneration(self):
        client = _client_with_scores(side_effect=RuntimeError("judge down"))
        regen_calls = {"n": 0}

        async def regenerate_fn(facts, style, settings):
            regen_calls["n"] += 1
            return StyledCaption(style=style, text="regenerated")

        with mock.patch("openai.AsyncOpenAI", return_value=client):
            result = asyncio.run(
                judge_and_regenerate(
                    facts=FACTS,
                    captions={"formal": "original text"},
                    settings=_settings(),
                    regenerate_fn=regenerate_fn,
                    max_retries=2,
                )
            )

        # Neutral fallback scores are maximal, so regeneration should
        # never be triggered by a judge outage.
        assert regen_calls["n"] == 0
        assert result["formal"] == "original text"
