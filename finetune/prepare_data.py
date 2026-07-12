"""Prompt 8B (optional, experimental): build a training set of
(GroundingFacts, style) -> caption pairs for the facts -> styled-caption
step only (not perception/grounding itself).

This script does NOT touch src/ — it imports src.styling/src.schemas/
src.config as a read-only consumer, exactly like any other client of those
modules, so it stays a fully isolated experiment per the plan's Phase 5
note ("does not silently eat the whole budget").

Labels come from `styling.style_caption`, i.e. a strong "teacher" model
(settings.styling_model) generating example captions for a seed set of
GroundingFacts, across all four styles. The seed facts are deliberately
generic (not tied to the project's known example clips) so a model
fine-tuned on this data doesn't overfit to a handful of specific scenes.

Usage:
    # Real mode: calls the configured teacher model (needs
    # GROUNDING_MODEL/STYLING_MODEL/JUDGE_MODEL env vars + network access).
    python -m finetune.prepare_data --out finetune/data/train.jsonl

    # Dry-run mode: no network/API calls at all. Produces a schema-valid
    # training set using deterministic placeholder captions, purely to
    # smoke-test the data pipeline shape (seed facts -> styles -> JSONL)
    # without needing credentials. Useful for CI / a bare dev box.
    python -m finetune.prepare_data --out finetune/data/train_dryrun.jsonl --dry-run

Output format: one JSON object per line:
    {"facts": {...GroundingFacts fields...}, "style": "formal", "caption": "..."}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow running as `python -m finetune.prepare_data` from the repo root
# without requiring the package to be pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, SettingsError, load_settings  # noqa: E402
from src.schemas import ALL_STYLES, GroundingFacts, Style  # noqa: E402
from src.styling import style_caption  # noqa: E402

logger = logging.getLogger(__name__)

# Generic, non-clip-specific seed facts. Intentionally varied in subject,
# setting, and mood so the resulting captions cover a spread of scenarios
# rather than overfitting to any one scene (per the plan's Phase 6
# generalization note, applied here to training-data generation too).
SEED_FACTS: list[GroundingFacts] = [
    GroundingFacts(
        subjects=["a golden retriever"], actions=["running", "chasing a ball"],
        setting="a sunny park", mood="playful",
        on_screen_text=[], audible_speech=None, notable_sounds=["barking"],
    ),
    GroundingFacts(
        subjects=["two chefs"], actions=["chopping vegetables", "plating a dish"],
        setting="a restaurant kitchen", mood="focused",
        on_screen_text=[], audible_speech="Careful, that pan is hot.",
        notable_sounds=["sizzling"],
    ),
    GroundingFacts(
        subjects=["a commuter"], actions=["waiting", "checking a phone"],
        setting="a rainy train platform", mood="tired",
        on_screen_text=["PLATFORM 2 DELAYED"], audible_speech=None,
        notable_sounds=["rain", "announcement chime"],
    ),
    GroundingFacts(
        subjects=["a software engineer"], actions=["typing", "staring at an error"],
        setting="a home office at night", mood="frustrated",
        on_screen_text=["Build failed"], audible_speech=None, notable_sounds=[],
    ),
    GroundingFacts(
        subjects=["a crowd", "a street performer"], actions=["juggling", "applauding"],
        setting="a busy city square", mood="festive",
        on_screen_text=[], audible_speech=None, notable_sounds=["cheering", "music"],
    ),
    GroundingFacts(
        subjects=["a toddler", "a puppy"], actions=["playing on a rug"],
        setting="a living room", mood="lighthearted",
        on_screen_text=[], audible_speech=None, notable_sounds=["giggling"],
    ),
    GroundingFacts(
        subjects=["a hiker"], actions=["climbing a rocky trail"],
        setting="a foggy mountainside", mood="determined",
        on_screen_text=[], audible_speech=None, notable_sounds=["wind"],
    ),
    GroundingFacts(
        subjects=["a cashier", "a customer"], actions=["scanning items", "paying"],
        setting="a small grocery store", mood="mundane",
        on_screen_text=["Total: $14.32"], audible_speech="Do you have a rewards card?",
        notable_sounds=["beep"],
    ),
]


def _dry_run_caption(facts: GroundingFacts, style: Style) -> str:
    """Deterministic, offline placeholder caption used only in --dry-run
    mode, so the data-shape can be smoke-tested with no network/API calls.
    Not a real training label — never used unless --dry-run is passed."""
    subject = facts.subjects[0] if facts.subjects else "something"
    return f"[{style}] {subject} in {facts.setting or 'an unknown setting'}."


async def _generate_examples(
    seed_facts: list[GroundingFacts],
    styles: list[Style],
    settings: Settings | None,
    dry_run: bool,
) -> list[dict]:
    examples: list[dict] = []
    for facts in seed_facts:
        for style in styles:
            if dry_run:
                caption_text = _dry_run_caption(facts, style)
            else:
                assert settings is not None  # guaranteed by caller in real mode
                styled = await style_caption(facts, style, settings)
                caption_text = styled.text
                if caption_text == settings.fallback_caption:
                    logger.warning(
                        "teacher call fell back to fallback_caption for "
                        "setting=%r style=%r — skipping this example rather "
                        "than training on a non-informative label",
                        facts.setting, style,
                    )
                    continue
            examples.append(
                {
                    "facts": facts.model_dump(mode="json"),
                    "style": style,
                    "caption": caption_text,
                }
            )
    return examples


def _write_jsonl(examples: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("finetune/data/train.jsonl"),
        help="Output JSONL path (default: finetune/data/train.jsonl)",
    )
    parser.add_argument(
        "--styles", nargs="+", default=list(ALL_STYLES),
        choices=list(ALL_STYLES),
        help="Which styles to generate examples for (default: all four)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Skip all network/API calls and use deterministic placeholder "
            "captions instead, purely to smoke-test the data pipeline shape."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    settings: Settings | None = None
    if not args.dry_run:
        try:
            settings = load_settings()
        except SettingsError as exc:
            print(f"[prepare_data] FATAL: {exc}", file=sys.stderr)
            print(
                "[prepare_data] hint: pass --dry-run to smoke-test the data "
                "pipeline shape without needing model credentials",
                file=sys.stderr,
            )
            return 1

    examples = asyncio.run(
        _generate_examples(SEED_FACTS, list(args.styles), settings, args.dry_run)
    )

    if not examples:
        print("[prepare_data] FATAL: no examples were generated", file=sys.stderr)
        return 1

    _write_jsonl(examples, args.out)
    print(f"[prepare_data] wrote {len(examples)} examples to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
