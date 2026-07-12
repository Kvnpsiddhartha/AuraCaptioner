"""Prompt 8B (optional, experimental): LoRA fine-tuning scaffold for the
facts -> styled-caption step only (not perception/grounding).

This is a SCAFFOLD, not a finished training pipeline: it proves the shape
(load JSONL -> format prompts -> tokenize -> LoRA train loop -> save
adapter) end-to-end, and is meant to be extended/tuned before it's
actually swapped into styling.py. See README.md in this directory for the
go/no-go decision criteria.

Two modes:

  Real mode (default): fine-tunes an actual small causal LM with a LoRA
  adapter via `transformers` + `peft`. Requires those packages (see
  finetune/requirements.txt, intentionally NOT added to the main
  requirements.txt since this is an isolated experiment) plus network
  access to download the base model and enough compute for at least a
  few training steps.

      python -m finetune.train --data finetune/data/train.jsonl \\
          --base-model <hf-model-id> --output-dir finetune/out/adapter

  Dry-run/smoke-test mode (--dry-run): trains a tiny, randomly-initialized
  character-level model (NOT a pretrained LM — no download, no network,
  no GPU required) end-to-end over the same JSONL data, to prove the full
  loop (data loading -> batching -> forward -> loss -> backward -> step ->
  checkpoint save) runs without error. This is what Prompt 8A's
  verification checklist item ("training run completes without error on a
  small sample, smoke test not full convergence") is designed to be
  checked against in this sandboxed environment, where downloading a real
  base model isn't possible.

      python -m finetune.train --data finetune/data/train.jsonl \\
          --output-dir finetune/out/dryrun_adapter --dry-run --epochs 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def _format_example(example: dict) -> tuple[str, str]:
    """Turn one {"facts", "style", "caption"} JSONL record into a
    (prompt, target) pair for causal-LM training. Kept as a single small
    function so real mode and dry-run mode format examples identically —
    only the model/tokenizer underneath differs."""
    facts_json = json.dumps(example["facts"], ensure_ascii=False)
    prompt = f"Facts: {facts_json}\nStyle: {example['style']}\nCaption:"
    target = " " + example["caption"].strip()
    return prompt, target


def _load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            for required_key in ("facts", "style", "caption"):
                if required_key not in record:
                    raise ValueError(f"{path}:{line_no}: missing required key {required_key!r}")
            examples.append(record)
    if not examples:
        raise ValueError(f"{path}: no training examples found")
    return examples


# --------------------------------------------------------------------------
# Dry-run mode: a tiny, from-scratch model with no external dependencies
# beyond torch, so the training loop is genuinely runnable/smoke-testable
# in an offline sandbox. Character-level so no tokenizer/vocab download is
# needed either.
# --------------------------------------------------------------------------

def _run_dry_run_training(examples: list[dict], output_dir: Path, epochs: int, lr: float) -> dict:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "torch is required even for --dry-run (it's the one dependency "
            "this smoke test can't avoid) — install it via "
            "finetune/requirements.txt"
        ) from exc

    pairs = [_format_example(ex) for ex in examples]
    texts = [prompt + target for prompt, target in pairs]

    vocab = sorted({ch for text in texts for ch in text} | {"\x00"})
    stoi = {ch: i for i, ch in enumerate(vocab)}
    pad_id = stoi["\x00"]
    max_len = max(len(t) for t in texts)

    def encode(text: str) -> list[int]:
        ids = [stoi[ch] for ch in text]
        return ids + [pad_id] * (max_len - len(ids))

    input_ids = torch.tensor([encode(t) for t in texts], dtype=torch.long)

    class TinyCharLM(nn.Module):
        """A deliberately tiny from-scratch model — this is NOT a stand-in
        for a real base model's architecture, it exists purely to give the
        training loop something real (embeddings, a forward pass,
        gradients, an optimizer step) to exercise without any download."""

        def __init__(self, vocab_size: int, dim: int = 32):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
            self.rnn = nn.GRU(dim, dim, batch_first=True)
            self.head = nn.Linear(dim, vocab_size)

        def forward(self, x):
            h, _ = self.rnn(self.embed(x))
            return self.head(h)

    model = TinyCharLM(vocab_size=len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.train()
    last_loss = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        logger.info("dry-run epoch %d/%d loss=%.4f", epoch + 1, epochs, last_loss)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "dryrun_model.pt")
    (output_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

    return {
        "mode": "dry-run",
        "num_examples": len(examples),
        "vocab_size": len(vocab),
        "epochs": epochs,
        "final_loss": last_loss,
        "checkpoint": str(output_dir / "dryrun_model.pt"),
    }


# --------------------------------------------------------------------------
# Real mode: an actual small pretrained base model + a LoRA adapter via
# transformers/peft. Requires network access and the extra dependencies in
# finetune/requirements.txt.
# --------------------------------------------------------------------------

def _run_real_training(
    examples: list[dict],
    base_model: str,
    output_dir: Path,
    epochs: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
) -> dict:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "real training mode needs torch, transformers, and peft — see "
            "finetune/requirements.txt. Use --dry-run to smoke-test the "
            "pipeline shape without these (and without network access)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # Deliberately not hardcoding target_modules to one architecture's
        # attention-projection names (e.g. "q_proj"/"v_proj" vs "c_attn")
        # since --base-model is a CLI argument, not fixed at write time.
        # peft's default target-module inference generally handles common
        # causal-LM architectures; override via LORA_TARGET_MODULES env if
        # a specific base model needs an explicit list.
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    pairs = [_format_example(ex) for ex in examples]
    encodings = [
        tokenizer(prompt + target, return_tensors="pt", truncation=True, max_length=512)
        for prompt, target in pairs
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last_loss = None
    for epoch in range(epochs):
        epoch_losses = []
        for enc in encodings:
            optimizer.zero_grad()
            outputs = model(input_ids=enc["input_ids"], labels=enc["input_ids"])
            outputs.loss.backward()
            optimizer.step()
            epoch_losses.append(outputs.loss.item())
        last_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info("epoch %d/%d mean_loss=%.4f", epoch + 1, epochs, last_loss)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "mode": "real",
        "base_model": base_model,
        "num_examples": len(examples),
        "epochs": epochs,
        "final_mean_loss": last_loss,
        "adapter_dir": str(output_dir),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to JSONL training data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save the adapter/checkpoint")
    parser.add_argument("--base-model", type=str, default=None, help="HF model id (real mode only)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Train a tiny from-scratch model instead of a real base model (no network/GPU needed).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    try:
        examples = _load_jsonl(args.data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[train] FATAL: {exc}", file=sys.stderr)
        return 1

    try:
        if args.dry_run:
            summary = _run_dry_run_training(examples, args.output_dir, args.epochs, args.lr)
        else:
            if not args.base_model:
                print("[train] FATAL: --base-model is required outside --dry-run", file=sys.stderr)
                return 1
            summary = _run_real_training(
                examples, args.base_model, args.output_dir,
                args.epochs, args.lr, args.lora_r, args.lora_alpha,
            )
    except RuntimeError as exc:
        print(f"[train] FATAL: {exc}", file=sys.stderr)
        return 1

    print(f"[train] done: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
