"""Tests for src/ocr.py — cheap text-region gate + OCR extraction."""
from __future__ import annotations

from unittest import mock

import pytest

import src.ocr as ocr_module
from src.ocr import extract_text, has_probable_text

pytest.importorskip("cv2", reason="opencv not installed")


class TestHasProbableText:
    def test_true_for_frame_with_clear_text(self, text_frame):
        assert has_probable_text([text_frame]) is True

    def test_false_for_blank_frame(self, blank_frame):
        assert has_probable_text([blank_frame]) is False

    def test_false_for_empty_frame_list(self):
        assert has_probable_text([]) is False

    def test_false_on_unreadable_frame_never_raises(self, tmp_path):
        garbage = tmp_path / "not-an-image.jpg"
        garbage.write_bytes(b"definitely not a jpeg")
        assert has_probable_text([garbage]) is False


class TestExtractText:
    def test_returns_empty_and_skips_ocr_when_gate_is_false(self, blank_frame):
        with mock.patch.object(ocr_module, "has_probable_text", return_value=False):
            with mock.patch("rapidocr_onnxruntime.RapidOCR") as engine_cls:
                result = extract_text([blank_frame])
        assert result == []
        engine_cls.assert_not_called()

    def test_dedupes_and_strips_empty_strings(self, text_frame):
        fake_engine = mock.Mock()
        fake_engine.return_value = (
            [
                (None, "Hello World", 0.99),
                (None, "Hello World", 0.98),  # exact duplicate
                (None, "  ", 0.5),  # whitespace-only, must be dropped
                (None, "", 0.4),  # empty, must be dropped
                (None, "Second Line", 0.95),
            ],
            0.01,
        )
        with mock.patch.object(ocr_module, "has_probable_text", return_value=True):
            with mock.patch(
                "rapidocr_onnxruntime.RapidOCR", return_value=fake_engine
            ):
                result = extract_text([text_frame])

        assert result == ["Hello World", "Second Line"]
        assert len(result) == len(set(result))

    def test_returns_empty_on_ocr_engine_construction_failure(self, text_frame):
        with mock.patch.object(ocr_module, "has_probable_text", return_value=True):
            with mock.patch(
                "rapidocr_onnxruntime.RapidOCR", side_effect=RuntimeError("model load failed")
            ):
                result = extract_text([text_frame])
        assert result == []

    def test_never_raises_out_of_function(self, text_frame):
        with mock.patch.object(ocr_module, "has_probable_text", side_effect=Exception("boom")):
            # has_probable_text itself guarantees no-raise, but extract_text
            # must never blow up even if that contract were ever violated
            # upstream in a refactor — regression guard.
            try:
                extract_text([text_frame])
            except Exception:
                pytest.fail("extract_text must never raise")
