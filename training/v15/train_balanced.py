#!/usr/bin/env python3
"""Independent balanced SFT trainer for StoryAI V15.

Run from repository root:
    python -m training.v15.train_balanced --help

Does not modify training/v15/train.py. Uses the original GPT model and the
original V15 get_sft_batch implementation, and trains equal *per-task losses*.
"""
import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch

from training.v15.model import GPT, GPTConfig
from training.v15.train import get_sft_batch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path, vocab, block_size, check_all):
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid SFT data: {path}")
    check = rows if check_all else rows[:min(32, len(rows))]
    for row in check:
        ids, labels = row["input_ids"], row["labels"]
        if len(ids) != len(labels) or len(ids) < 2:
            raise ValueError(f"Malformed SFT row: {path}")
        if not any(int(v) >= 0 for v in labels[1:]):
            raise ValueError(f"No completion targets: {path}")
        if any(int(v) < 0 or int(v) >= vocab for v in ids):
            raise ValueError(f"Invalid token ID: {path}")
        if any(int(v) not in (-1, -100) and not (0 <= int(v) < vocab) for v in labels):
            raise ValueError(f"Invalid label: {path}")
        # Ensure the completion doesn't get entirely cropped away.
        if len(ids) > block_size + 1 and not any(int(v) >= 0 for v in labels[-block_size:]):
            raise ValueError(f"Completion lost by cropping: {path}")
    return rows


def normalize_labels(rows):
    # The original trainer ignores -100 and pads with -1.
    # V2 synthetic data sometimes uses -1 as prompt mask; standardize it.
    for row in rows:
        labels = row["labels"]
        if (labels == -1).any():
            labels = labels.clone()
            labels[labels == -1] = -100
            row["labels"] = labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="dataset/v15_prose_expanded")
    ap.add_argument("--sft-dir", default="dataset/v15_prose_expanded/continuation_sft")
    ap.add_argument("--reasoning-sft-dir", default="dataset/v15_prose_expanded/reasoning_v2_sft")
    ap.add_argument("--init-from", help="Trusted V15 pretrain or SFT checkpoint; model weights only")
    ap.add_argument("--out-dir", default="checkpoints/v15_balanced")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval-seed", type=int, default=8675309)
    ap.add_argument("--weight-decay", type=float, default=.1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--smoke-test", action="store_true", help="One step, no checkpoint written; small model if --init-from omitted")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate a checkpoint without writing or training")
    ap.add_argument("--story-weight", type=float, default=0.5, help="Story loss weight in [0,1]; reasoning weight is the remainder")
    a = ap.parse_args()

    if not 0 <= a.story_weight <= 1:
        ap.error("--story-weight must lie between 0 and 1")
    if a.eval_only and a.smoke_test:
        ap.error("--eval-only and --smoke-test are mutually exclusive")
    if min(a.max_steps, a.batch_size, a.grad_accum, a.block_size, a.eval_every, a.eval_batches, a.save_every) < 1:
        ap.error("Steps, batch size, accumulation, block size and evaluation intervals must be positive.")
    if a.warmup_steps < 0 or not 0 < a.min_lr <= a.lr:
        ap.error("Require warmup >= 0 and 0 < min-lr <= lr")
    if not a.smoke_test and not a.init_from:
        ap.error("--init-from is required for actual training")
    if a.smoke_test:
        a.max_steps, a.batch_size, a.grad_accum = 1, 1, 1
        a.eval_every = a.save_every = a.eval_batches = 1
        if not a.init_from:
            a.block_size = min(a.block_size, 128)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(a.seed)

    data = Path(a.data_dir)
    report = json.loads((data / "build_report.json").read_text(encoding="utf-8-sig"))
    tok_hash = sha256(data / "tokenizer.json")
    if report["tokenizer_sha256"] != tok_hash:
        raise ValueError("Dataset tokenizer/report SHA256 mismatch")
    vocab = int(report["vocab_size"])

    ckpt = None
    if a.init_from:
        ckpt = torch.load(a.init_from, map_location="cpu", weights_only=False)
        if ckpt.get("mode") not in ("pretrain", "sft"):
            raise ValueError("Unknown checkpoint mode: expected pretrain or sft")
        if ckpt.get("tokenizer_sha256") != tok_hash:
            raise ValueError("Checkpoint tokenizer SHA256 mismatch")
        cfg = GPTConfig(**ckpt["model_config"])
        if cfg.vocab_size != vocab or cfg.block_size != a.block_size:
            raise ValueError("Checkpoint vocabulary/context mismatch")
    else:
        cfg = GPTConfig(block_size=a.block_size, vocab_size=vocab, n_layer=2, n_head=4, n_embd=128)

    dirs = {"story": Path(a.sft_dir), "reasoning": Path(a.reasoning_sft_dir)}
    rows = {
        task: {
            split: load_rows(path / (split + "_sft.pt"), vocab, cfg.block_size, not a.smoke_test)
            for split in ("train", "val")
        }
        for task, path in dirs.items()
    }
    for by_split in rows.values():
        for r in by_split.values():
            normalize_labels(r)
    print("Dataset sizes:", {task: {split: len(part) for split, part in splits.items()}
                             for task, splits in rows.items()}, flush=True)

    model = GPT(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    autocast = lambda: torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

    def batch(task, split):
        return get_sft_batch(rows[task][split], a.batch_size, cfg.block_size, device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        # Fixed RNG for fair comparisons across checkpoints.
        py_state = random.getstate()
        torch_state = torch.get_rng_state()
        random.seed(a.eval_seed)
        torch.manual_seed(a.eval_seed)
        results = {}
        try:
            for task in ("story", "reasoning"):
                values = []
                for _ in range(a.eval_batches):
                    x, y = batch(task, "val")
                    with autocast():
                        _, loss = model(x, y)
                    values.append(float(loss))
                results[task] = sum(values) / len(values)
            results["balanced"] = a.story_weight * results["story"] + (1 - a.story_weight) * results["reasoning"]
        finally:
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
            model.train()
        return results

    print(f"Device={device}, story_weight={a.story_weight}, reasoning_weight={1-a.story_weight}, AMP={amp_dtype if use_amp else 'off'}", flush=True)
    baseline = evaluate()
    print("BASELINE_FIXED:", json.dumps(baseline), flush=True)
    if a.eval_only:
        return
    best = baseline["balanced"]
    out_check = Path(a.out_dir)
    if any(out_check.glob("*.pt")):
        raise FileExistsError(f"Refusing to overwrite checkpoints in {out_check}; choose a fresh --out-dir")

    for step in range(1, a.max_steps + 1):
        if a.warmup_steps and step <= a.warmup_steps:
            lr = a.lr * step / a.warmup_steps
        else:
            fraction = min(1.0, max(0.0, (step - a.warmup_steps) /
                                     max(1, a.max_steps - a.warmup_steps)))
            lr = a.min_lr + .5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * fraction))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        sums = {"story": 0., "reasoning": 0.}
        model.train()
        for _ in range(a.grad_accum):
            x_s, y_s = batch("story", "train")
            x_r, y_r = batch("reasoning", "train")
            with autocast():
                _, ls = model(x_s, y_s)
                _, lr_loss = model(x_r, y_r)
                loss = a.story_weight * ls + (1 - a.story_weight) * lr_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            sums["story"] += float(ls.detach()) / a.grad_accum
            sums["reasoning"] += float(lr_loss.detach()) / a.grad_accum
            scaler.scale(loss / a.grad_accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        print(f"step={step} lr={lr:.3g} story_loss={sums['story']:.4f} "
              f"reasoning_loss={sums['reasoning']:.4f}", flush=True)

        need_eval = step % a.eval_every == 0 or step == a.max_steps
        need_save = step % a.save_every == 0 or step == a.max_steps
        metrics = evaluate() if need_eval else None
        if metrics is not None:
            print("eval:", json.dumps(metrics), flush=True)
        if a.smoke_test:
            continue
        if need_save or (metrics is not None and metrics["balanced"] < best):
            out = Path(a.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            state = {
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(), "model_config": vars(cfg).copy(),
                "mode": "sft", "step": step, "tokenizer_sha256": tok_hash,
                "loss_weights": {"story": a.story_weight, "reasoning": 1-a.story_weight},
                "metrics": metrics,
            }
            if need_save:
                torch.save(state, out / "sft_latest.pt")
            if metrics is not None and metrics["balanced"] < best:
                best = metrics["balanced"]
                torch.save(state, out / "sft_best.pt")
    if a.smoke_test:
        print("PASS: balanced forward/backward optimizer step; no checkpoints written")


if __name__ == "__main__":
    main()
