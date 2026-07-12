"""On-screen text: a cheap gate followed by real OCR.

`has_probable_text` must be cheap — a fast text-region candidate detector
(MSER), not full OCR — so plain nature/animal/no-text footage never pays
for a real OCR pass. `extract_text` re-uses that gate internally.

Takes `frames: list[Path]` directly (plain image paths) rather than
importing sampling.py, so this module stays independently testable.
"""
from __future__ import annotations

from pathlib import Path

# MSER candidate-region filter: characters in on-screen text/signage at
# typical video resolutions land in roughly this height range, with a
# bounded width/height ratio (excludes long thin edges/lines and large
# blob-like regions that aren't text strokes).
MSER_MIN_REGION_HEIGHT = 8
MSER_MAX_REGION_HEIGHT = 120
MSER_MAX_ASPECT_RATIO = 8.0
# Number of plausible text-region candidates (summed across all sampled
# frames) required before we consider full OCR worth running.
MIN_TEXT_REGION_CANDIDATES = 6


def _text_region_candidate_count(gray_image) -> int:
    import cv2

    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray_image)
    count = 0
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        if h < MSER_MIN_REGION_HEIGHT or h > MSER_MAX_REGION_HEIGHT:
            continue
        aspect = w / h if h else 0
        if aspect > MSER_MAX_ASPECT_RATIO:
            continue
        count += 1
    return count


def has_probable_text(frames: list[Path]) -> bool:
    """Cheap heuristic: True if enough MSER text-like region candidates are
    found across `frames` to justify running full OCR.

    Returns False on any failure (missing OpenCV, unreadable image, ...) —
    skipping OCR is always the safe default.
    """
    if not frames:
        return False
    try:
        import cv2
    except Exception:
        return False

    total_candidates = 0
    for frame_path in frames:
        try:
            gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            total_candidates += _text_region_candidate_count(gray)
        except Exception:
            continue
        if total_candidates >= MIN_TEXT_REGION_CANDIDATES:
            return True

    return total_candidates >= MIN_TEXT_REGION_CANDIDATES


def _clean_dedupe(strings: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in strings:
        text = raw.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def extract_text(frames: list[Path]) -> list[str]:
    """Run OCR across `frames` and return deduplicated, cleaned text
    strings found.

    Calls has_probable_text() first and returns [] immediately (without
    invoking OCR) if it's False. Returns [] on any failure rather than
    raising — on-screen text is supplementary context, never a blocker.
    This function is wrapped end-to-end as a defense-in-depth safety net,
    on top of has_probable_text's own internal guarantee not to raise.
    """
    try:
        if not has_probable_text(frames):
            return []

        try:
            from rapidocr_onnxruntime import RapidOCR  # optional heavy dependency
        except Exception:
            return []

        try:
            engine = RapidOCR()
        except Exception:
            return []

        found: list[str] = []
        for frame_path in frames:
            try:
                result, _elapsed = engine(str(frame_path))
            except Exception:
                continue
            if not result:
                continue
            for _box, text, _score in result:
                if text:
                    found.append(text)

        return _clean_dedupe(found)
    except Exception:
        return []
