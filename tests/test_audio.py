"""Tests for src/audio.py — VAD gate + transcription."""
from __future__ import annotations

from unittest import mock

import pytest

import src.audio as audio_module
from src.audio import has_speech, transcribe

pytest.importorskip("webrtcvad", reason="webrtcvad not installed")


class TestHasSpeech:
    def test_true_for_clip_with_clear_speech(self, speech_video):
        assert has_speech(speech_video) is True

    def test_false_for_silent_clip(self, static_silent_video):
        assert has_speech(static_silent_video) is False

    def test_false_on_missing_file_never_raises(self, tmp_path):
        assert has_speech(tmp_path / "does-not-exist.mp4") is False

    def test_false_on_corrupt_file_never_raises(self, tmp_path):
        garbage = tmp_path / "garbage.mp4"
        garbage.write_bytes(b"not a real video")
        assert has_speech(garbage) is False


class TestTranscribe:
    def test_returns_none_and_skips_model_when_no_speech(self, static_silent_video):
        with mock.patch.object(audio_module, "_load_whisper_model") as loader:
            result = transcribe(static_silent_video)
        assert result is None
        loader.assert_not_called()

    def test_returns_none_when_model_call_fails(self, speech_video):
        with mock.patch.object(
            audio_module, "_load_whisper_model", side_effect=RuntimeError("boom")
        ):
            result = transcribe(speech_video)
        assert result is None

    def test_returns_transcript_text_on_success(self, speech_video):
        fake_model = mock.Mock()
        fake_segment = mock.Mock()
        fake_segment.text = " hello there "
        fake_model.transcribe.return_value = ([fake_segment], mock.Mock())

        with mock.patch.object(audio_module, "_load_whisper_model", return_value=fake_model):
            result = transcribe(speech_video)

        assert result == "hello there"

    def test_never_raises_out_of_function(self, speech_video):
        with mock.patch.object(
            audio_module, "_load_whisper_model", side_effect=Exception("catastrophic")
        ):
            # Must not propagate — always degrade to None.
            assert transcribe(speech_video) is None
