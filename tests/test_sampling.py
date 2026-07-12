"""Tests for src/sampling.py — frame count scaling, uniform + keyframe sampling."""
from __future__ import annotations

from pathlib import Path

import cv2

from src.sampling import (
    frame_count_for_duration,
    sample_keyframes,
    sample_uniform,
)


def _is_valid_image(path: Path) -> bool:
    img = cv2.imread(str(path))
    return img is not None and img.size > 0


class TestFrameCountForDuration:
    def test_both_within_clamp_range(self):
        short = frame_count_for_duration(30, fmin=8, fmax=20)
        long = frame_count_for_duration(120, fmin=8, fmax=20)
        assert 8 <= short <= 20
        assert 8 <= long <= 20

    def test_longer_clip_gets_at_least_as_many_frames(self):
        short = frame_count_for_duration(30, fmin=8, fmax=20)
        long = frame_count_for_duration(120, fmin=8, fmax=20)
        assert long >= short

    def test_clamped_at_fmin_for_very_short_clips(self):
        assert frame_count_for_duration(0.001, fmin=8, fmax=20) == 8

    def test_clamped_at_fmax_for_very_long_clips(self):
        assert frame_count_for_duration(10_000, fmin=8, fmax=20) == 20

    def test_zero_or_negative_duration_returns_fmin(self):
        assert frame_count_for_duration(0, fmin=8, fmax=20) == 8
        assert frame_count_for_duration(-5, fmin=8, fmax=20) == 8


class TestSampleUniform:
    def test_returns_exactly_n_valid_frames(self, multi_cut_video, tmp_path):
        frames = sample_uniform(multi_cut_video, 6, tmp_path)
        assert len(frames) == 6
        for frame in frames:
            assert frame.exists()
            assert _is_valid_image(frame)

    def test_frames_are_sorted(self, multi_cut_video, tmp_path):
        frames = sample_uniform(multi_cut_video, 5, tmp_path)
        assert frames == sorted(frames)

    def test_n_zero_returns_empty(self, multi_cut_video, tmp_path):
        assert sample_uniform(multi_cut_video, 0, tmp_path) == []

    def test_concurrent_calls_do_not_collide(self, multi_cut_video, tmp_path):
        first = sample_uniform(multi_cut_video, 3, tmp_path)
        second = sample_uniform(multi_cut_video, 3, tmp_path)
        assert first[0].parent != second[0].parent
        assert all(f.exists() for f in first)
        assert all(f.exists() for f in second)


class TestSampleKeyframes:
    def test_uses_real_scene_detection_when_enough_cuts_exist(self, multi_cut_video, tmp_path):
        # multi_cut_video has 3 hard cuts; asking for <= 3 frames should be
        # satisfiable by real scene detection, not the uniform fallback.
        frames = sample_keyframes(multi_cut_video, 3, tmp_path)
        assert len(frames) == 3
        assert frames[0].parent.name.startswith("keyframes_")
        for frame in frames:
            assert _is_valid_image(frame)

    def test_falls_back_to_uniform_on_static_clip(self, static_silent_video, tmp_path):
        # No scene changes at all in a single-color clip -> must still
        # return the requested frame count via the uniform fallback.
        frames = sample_keyframes(static_silent_video, 5, tmp_path)
        assert len(frames) == 5
        for frame in frames:
            assert frame.exists()
            assert _is_valid_image(frame)

    def test_n_zero_returns_empty(self, multi_cut_video, tmp_path):
        assert sample_keyframes(multi_cut_video, 0, tmp_path) == []
