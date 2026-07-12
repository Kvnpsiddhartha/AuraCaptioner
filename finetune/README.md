# Fine-tuning scaffold (Prompt 8B, optional/experimental)

Purpose: explore replacing the **facts → styled-caption** LLM call in
`src/styling.py` with a small fine-tuned model, to cut latency/cost per
style. This directory is intentionally isolated — it does not import from
or modify anything in `src/`, and nothing here is wired into the real
pipeline. Perception (grounding/verification) is out of scope: this only
targets the styling step, which is a much simpler text→text mapping.

## Files

- `prepare_data.py` — builds a JSONL training set of
  `{"facts": ..., "style": ..., "caption": ...}` records. Labels come from
  `src.styling.style_caption`, i.e. sampling a strong "teacher" model
  across a generic seed set of `GroundingFacts` and all four styles.
- `train.py` — a LoRA training loop over that JSONL data.
- `requirements.txt` — extra deps for this directory only (torch,
  transformers, peft, accelerate) — never merged into the main image.

## Two modes, in both scripts

| Mode | What it does | Requirements |
|---|---|---|
| Real (default) | Calls the real teacher model (`prepare_data.py`) / fine-tunes a real HF base model with LoRA (`train.py`) | API credentials + network access (`prepare_data.py`); network + compute to download and train a base model (`train.py`) |
| `--dry-run` | `prepare_data.py`: deterministic placeholder captions, no API calls. `train.py`: trains a tiny from-scratch character-level model, no download, no GPU | None beyond `torch` for `train.py --dry-run` |

`--dry-run` exists specifically so the data → train → checkpoint shape can
be smoke-tested in an offline/sandboxed environment (like the one this
scaffold was originally built in) where downloading a real base model
isn't possible. It is **not** a substitute for a real training run and its
output should never be evaluated for caption quality.

## Quickstart

```bash
pip install -r finetune/requirements.txt   # or just `torch` for --dry-run

# 1. Build a (smoke-test) training set with no API calls:
python -m finetune.prepare_data --out finetune/data/train_dryrun.jsonl --dry-run

# 2. Smoke-test the training loop with no network/GPU:
python -m finetune.train --data finetune/data/train_dryrun.jsonl \
    --output-dir finetune/out/dryrun_adapter --dry-run --epochs 1

# --- once the above two commands both exit 0, the pipeline shape is
# --- proven and you're ready to try it for real:

# 3. Build a real training set (needs GROUNDING_MODEL/STYLING_MODEL/
#    JUDGE_MODEL env vars + network access — the teacher is settings.styling_model):
export GROUNDING_MODEL=... STYLING_MODEL=... JUDGE_MODEL=...
python -m finetune.prepare_data --out finetune/data/train.jsonl

# 4. Fine-tune a real small base model:
python -m finetune.train --data finetune/data/train.jsonl \
    --base-model <a small causal LM you have access to> \
    --output-dir finetune/out/adapter --epochs 3
```

## Time cost

- `prepare_data.py` (real mode): ~1 LLM call per (seed fact × style) pair.
  With 8 seed facts × 4 styles = 32 calls, at STYLING_API_TIMEOUT_SECONDS
  budget each, this is a few minutes at most, well inside a hackathon's
  exploration budget — but it's still API spend on top of the actual
  submission's runs, so treat it as a side experiment, not part of the
  timed pipeline run.
- `train.py` (real mode): highly dependent on the chosen base model size
  and available compute. A small (≤1-2B param) base model with LoRA on a
  single GPU for a few epochs over a few hundred examples is typically
  low-single-digit-hours, not something to attempt inside the 10-minute
  per-run container budget — this is offline prep work done well before
  the timed submission run, producing an adapter that would then need to
  be baked into the Docker image if adopted.

## Expected payoff

A fine-tuned small model *could* reduce per-style latency and API cost
versus calling a large hosted model four times per task (once per style).
Whether it's worth the offline time investment depends entirely on:
measured caption quality (accuracy + style_match, the same two axes
`src/judge.py` already scores) versus the current `styling.style_caption`
baseline, and how much of the remaining hackathon time budget is left to
spend on an optional experiment versus hardening the guaranteed-safe path.

## Go / no-go decision criteria

Do **not** swap this into `src/styling.py` unless ALL of the following
hold:

1. **Quality**: side-by-side `judge.judge_caption` scores (accuracy AND
   style_match) for the fine-tuned model are within a small, explicitly
   chosen tolerance of (or better than) the current teacher-model
   baseline, across a held-out sample of `GroundingFacts` not used in
   training.
2. **Latency**: measured per-call latency for the fine-tuned model
   (including any model-load overhead inside the container) is lower
   than, or at worst comparable to, the current `styling.style_caption`
   call — since latency reduction is the entire point of this experiment.
3. **Image size**: baking the base model + adapter into the Docker image
   keeps the final compressed image under the project's 10GB hard
   constraint (check via `scripts/check_image.sh`); if the base model
   alone is too large, this is an automatic no-go regardless of quality.
4. **Time remaining**: there is still enough hackathon time left to (a)
   integrate the fine-tuned model into `styling.py` behind the same
   never-raises/fallback contract every other module in `src/` follows,
   and (b) re-run the full Prompt 9 hardening verification checklist
   afterward. If time is short, keep the current teacher-model-backed
   `styling.py` — it is already correct, tested, and safe; this is a
   pure optimization, not a required feature.

If any criterion fails, stop here — the existing `styling.py` stays as
the shipped implementation.
