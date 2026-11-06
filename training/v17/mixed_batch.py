"""Single-pass mixed SFT sampling for Story AI v17."""

import random
import torch
from torch.nn.utils.rnn import pad_sequence


def get_mixed_sft_batch(
    prose_rows,
    grounded_rows,
    batch_size,
    block_size,
    device,
    grounded_ratio=0.30,
):
    if not 0 <= grounded_ratio <= 1:
        raise ValueError("Invalid grounded ratio")

    if not prose_rows or not grounded_rows:
        raise ValueError("Both datasets must be nonempty")

    xs, ys = [], []
    counts = {"prose": 0, "grounded": 0}

    for _ in range(batch_size):
        is_grounded = random.random() < grounded_ratio
        pool = grounded_rows if is_grounded else prose_rows
        row = random.choice(pool)

        counts["grounded" if is_grounded else "prose"] += 1

        ids = list(row["input_ids"])
        labels = list(row["labels"])

        if len(ids) != len(labels) or len(ids) < 2:
            raise ValueError("Invalid SFT record")

        if len(ids) > block_size + 1:
            supervised = [
                i for i, v in enumerate(labels)
                if int(v) != -100
            ]

            if not supervised:
                raise ValueError("No supervised tokens")

            boundary = supervised[0]
            start = max(0, boundary - max(1, block_size // 2))
            start = min(start, len(ids) - block_size - 1)

            ids = ids[start:start + block_size + 1]
            labels = labels[start:start + block_size + 1]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor([
            -1 if int(v) == -100 else int(v)
            for v in labels[1:]
        ], dtype=torch.long)

        if not (y != -1).any():
            raise ValueError("No supervised targets after cropping")

        xs.append(x)
        ys.append(y)

    x = pad_sequence(xs, batch_first=True, padding_value=0)
    y = pad_sequence(ys, batch_first=True, padding_value=-1)

    return x.to(device), y.to(device), counts
