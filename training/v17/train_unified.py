"""Story AI v17: 48.51M GPT foundation and unified SFT trainer.

Modes:
  pretrain : document-aware language-model training from scratch
  sft      : mixed prose, character and reasoning training

All trainable GPT weights are updated. No auxiliary pointer decoder.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from training.v15.model import GPT, GPTConfig
from training.v17.document_sampler import DocumentSampler


TASKS = ("prose", "character", "reasoning")
WEIGHTS = (0.50, 0.20, 0.30)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    p.add_argument("--data-dir", default="dataset/v15_prose_expanded")
    p.add_argument("--prose-dir",
                   default="dataset/v15_prose_expanded/continuation_sft")
    p.add_argument("--character-dir", default="dataset/v17_multi_event")
    p.add_argument("--reasoning-dir", default="dataset/v17_reasoning_planning")
    p.add_argument("--out-dir", default="checkpoints/v17")
    p.add_argument("--init-from")
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1704)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"],
                   default="auto")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def load_sft(directory, split):
    path = Path(directory) / f"{split}_sft.pt"
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def make_sft_batch(rows, indices, block_size, device):
    xs, ys = [], []

    for i in indices:
        row = rows[i]
        ids = list(row["input_ids"])
        labels = list(row["labels"])

        if len(ids) != len(labels) or len(ids) < 2:
            raise ValueError("Invalid SFT record")

        supervised = [
            j for j, value in enumerate(labels)
            if int(value) != -100
        ]
        if not supervised:
            raise ValueError("Unsupervised SFT record")

        if len(ids) > block_size + 1:
            boundary = supervised[0]
            start = max(0, boundary - max(1, block_size // 2))
            start = min(start, len(ids) - block_size - 1)
            ids = ids[start:start + block_size + 1]
            labels = labels[start:start + block_size + 1]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor([
            -1 if int(value) == -100 else int(value)
            for value in labels[1:]
        ], dtype=torch.long)

        if not (y != -1).any():
            raise ValueError("No targets after SFT cropping")

        xs.append(x)
        ys.append(y)

    return (
        pad_sequence(xs, batch_first=True, padding_value=0).to(device),
        pad_sequence(ys, batch_first=True, padding_value=-1).to(device),
    )


def main():
    a = arguments()

    if a.resume and a.init_from:
        raise ValueError("Choose --resume OR --init-from")
    if a.mode == "pretrain" and a.init_from:
        raise ValueError("Pretraining from scratch cannot use --init-from")
    if a.mode == "sft" and not (a.init_from or a.resume or a.smoke_test):
        raise ValueError("SFT requires foundation weights via --init-from")
    if min(a.steps, a.batch_size, a.grad_accum, a.block_size,
           a.eval_every, a.eval_batches, a.save_every) < 1:
        raise ValueError("Training dimensions and intervals must be positive")
    if not (0 < a.min_lr <= a.lr):
        raise ValueError("Require 0 < min-lr <= lr")
    if a.warmup < 0:
        raise ValueError("Warmup cannot be negative")

    if a.smoke_test:
        a.steps = 1
        a.batch_size = 1
        a.grad_accum = 1
        a.eval_batches = 1
        a.block_size = 128

    torch.set_num_threads(4)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    device = (
        "cuda" if a.device == "auto" and torch.cuda.is_available()
        else "cpu" if a.device == "auto"
        else a.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    root = Path(a.data_dir)
    tokenizer_path = root / "tokenizer.json"
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    config = GPTConfig(
        block_size=a.block_size,
        vocab_size=8192,
        n_layer=13,
        n_head=8,
        n_embd=512,
        dropout=0.0,
    )

    checkpoint = None
    if a.resume or a.init_from:
        checkpoint = torch.load(
            a.resume or a.init_from,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("tokenizer_sha256") != tokenizer_sha:
            raise ValueError("Checkpoint/tokenizer mismatch")
        if a.resume and checkpoint.get("mode") != a.mode:
            raise ValueError("Resume mode mismatch")
        if a.init_from and checkpoint.get("mode") != "pretrain":
            raise ValueError("SFT initialization must use a pretrain checkpoint")
        if checkpoint["model_config"] != vars(config):
            raise ValueError("Checkpoint/model configuration mismatch")

    model = GPT(config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr,
        weight_decay=0.1,
    )

    step = 0
    rng = random.Random(a.seed)
    task_counts = dict.fromkeys(TASKS, 0)

    if a.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        rng.setstate(checkpoint["sampling_rng"])

    # Use BF16 when available; otherwise CUDA FP16 with scaling.
    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if a.resume and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    if a.mode == "pretrain":
        train_tokens = np.load(root / "train.npy", mmap_mode="r")
        val_tokens = np.load(root / "val.npy", mmap_mode="r")

        train_sampler = DocumentSampler(
            train_tokens, eos_id=2,
            block_size=a.block_size, seed=a.seed,
        )

        if a.resume and "sampler_rng" in checkpoint:
            train_sampler.rng.bit_generator.state = checkpoint["sampler_rng"]

        def train_batch():
            return train_sampler.sample(a.batch_size, device)

        def evaluate():
            # Recreate sampler with fixed seed for reproducible validation.
            sampler = DocumentSampler(
                val_tokens, eos_id=2,
                block_size=a.block_size, seed=987654,
            )
            losses = []
            model.eval()
            with torch.no_grad():
                for _ in range(a.eval_batches):
                    x, y = sampler.sample(a.batch_size, device)
                    with torch.autocast(
                        device_type=device, dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        _, loss = model(x, y)
                    losses.append(float(loss.item()))
            model.train()
            return {"pretrain": sum(losses) / len(losses)}

    else:
        paths = {
            "prose": a.prose_dir,
            "character": a.character_dir,
            "reasoning": a.reasoning_dir,
        }
        datasets = {
            task: {
                split: load_sft(path, split)
                for split in ("train", "val")
            }
            for task, path in paths.items()
        }

        for task in TASKS:
            print(
                f"DATA {task}: "
                f"train={len(datasets[task]['train'])} "
                f"val={len(datasets[task]['val'])}",
                flush=True,
            )

        def train_batch():
            task = rng.choices(TASKS, weights=WEIGHTS, k=1)[0]
            rows = datasets[task]["train"]
            indices = [rng.randrange(len(rows)) for _ in range(a.batch_size)]
            task_counts[task] += len(indices)
            return make_sft_batch(rows, indices, a.block_size, device)

        def evaluate():
            scores = {}
            model.eval()
            with torch.no_grad():
                for task in TASKS:
                    rows = datasets[task]["val"]
                    local_rng = random.Random(987654)
                    losses = []
                    for _ in range(a.eval_batches):
                        indices = [
                            local_rng.randrange(len(rows))
                            for _ in range(a.batch_size)
                        ]
                        x, y = make_sft_batch(
                            rows, indices, a.block_size, device
                        )
                        with torch.autocast(
                            device_type=device, dtype=amp_dtype,
                            enabled=use_amp,
                        ):
                            _, loss = model(x, y)
                        losses.append(float(loss.item()))
                    scores[task] = sum(losses) / len(losses)
            model.train()
            return scores

    out = Path(a.out_dir)
    if not a.smoke_test:
        out.mkdir(parents=True, exist_ok=True)

    def save(name, metrics):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": vars(config),
            "tokenizer_sha256": tokenizer_sha,
            "mode": a.mode,
            "step": step,
            "metrics": metrics,
            "sampling_rng": rng.getstate(),
            "best_score": best_score,
            "prose_limit": prose_limit,
            "task_counts": task_counts.copy(),
            "torch_rng": torch.get_rng_state(),
        }
        if a.mode == "pretrain":
            state["sampler_rng"] = train_sampler.rng.bit_generator.state
        if device == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state()
        tmp = out / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(out / name)
        print("SAVED:", out / name, flush=True)

    if a.resume:
        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"])
        if device == "cuda" and "cuda_rng" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"])

    baseline = evaluate()
    print("BASELINE:", json.dumps(baseline), flush=True)

    # Keep the initial checkpoint available until a better one is verified.
    best_score = (
        baseline["pretrain"] if a.mode == "pretrain"
        else sum(baseline[t] * w for t, w in zip(TASKS, WEIGHTS))
    )
    prose_limit = baseline.get("prose", float("inf")) * 1.05

    if a.resume:
        best_path = out / (a.mode + "_best.pt")
        best_ckpt = (
            torch.load(best_path, map_location="cpu", weights_only=False)
            if best_path.exists() else checkpoint
        )

        def checkpoint_score(c):
            m = c.get("metrics", {})
            if a.mode == "pretrain":
                return float(m.get("pretrain", float("inf")))
            if all(t in m for t in TASKS):
                return sum(float(m[t]) * w for t, w in zip(TASKS, WEIGHTS))
            return float("inf")

        best_score = min(
            float(checkpoint.get("best_score", float("inf"))),
            float(best_ckpt.get("best_score", float("inf"))),
            checkpoint_score(best_ckpt),
        )

        prose_limit = float(
            checkpoint.get(
                "prose_limit",
                best_ckpt.get(
                    "prose_limit",
                    float(best_ckpt.get("metrics", {}).get(
                        "prose", baseline.get("prose", float("inf"))
                    )) * 1.05
                )
            )
        )
        task_counts.update(checkpoint.get("task_counts", {}))
        print(
            "RESUME BEST:", best_score,
            "PROSE LIMIT:", prose_limit,
            flush=True
        )

    if not a.smoke_test and not a.resume:
        save(a.mode + "_best.pt", baseline)

    model.train()

    while step < a.steps:
        optimizer.zero_grad(set_to_none=True)

        if step < a.warmup:
            lr = a.lr * (step + 1) / max(1, a.warmup)
        else:
            fraction = min(
                1.0,
                (step - a.warmup) / max(1, a.steps - a.warmup),
            )
            lr = a.min_lr + (a.lr - a.min_lr) * 0.5 * (
                1 + math.cos(math.pi * fraction)
            )

        for group in optimizer.param_groups:
            group["lr"] = lr

        train_losses = []

        for _ in range(a.grad_accum):
            x, y = train_batch()
            with torch.autocast(
                device_type=device, dtype=amp_dtype, enabled=use_amp
            ):
                _, loss = model(x, y)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")

            train_losses.append(float(loss.detach()))
            scaler.scale(loss / a.grad_accum).backward()

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if a.smoke_test or step % a.eval_every == 0 or step == a.steps:
            scores = evaluate()
            score = (
                scores["pretrain"] if a.mode == "pretrain"
                else sum(scores[t] * w for t, w in zip(TASKS, WEIGHTS))
            )

            print(
                "EVAL", step,
                json.dumps({
                    "lr": lr,
                    "train_loss": sum(train_losses) / len(train_losses),
                    "validation": scores,
                    "task_counts": task_counts,
                }),
                flush=True,
            )

            # Prose regression guard for the SFT stage.
            prose_ok = (
                a.mode == "pretrain"
                or scores["prose"] <= prose_limit
            )

            if not a.smoke_test and score < best_score and prose_ok:
                best_score = score
                save(a.mode + "_best.pt", scores)

        if (
            not a.smoke_test
            and (step % a.save_every == 0 or step == a.steps)
        ):
            save(a.mode + "_latest.pt", scores)

    if a.smoke_test:
        print("PASS: real GPT forward, backward, optimizer and evaluation")
        print("No checkpoint written.")


if __name__ == "__main__":
    main()
