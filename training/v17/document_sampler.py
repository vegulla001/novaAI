"""Document-aware next-token sampling for Story AI v17."""

import numpy as np
import torch


class DocumentSampler:
    def __init__(self, tokens, eos_id=2, block_size=1024, seed=42):
        self.tokens = tokens
        self.block_size = int(block_size)
        self.rng = np.random.default_rng(seed)

        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")

        ends = np.flatnonzero(tokens == eos_id)

        starts = np.concatenate((
            np.array([0], dtype=np.int64),
            ends[:-1] + 1
        ))

        stops = ends + 1

        # Include a final document if it has no EOS.
        if len(ends) == 0:
            starts = np.array([0], dtype=np.int64)
            stops = np.array([len(tokens)], dtype=np.int64)
        elif ends[-1] + 1 < len(tokens):
            starts = np.append(starts, ends[-1] + 1)
            stops = np.append(stops, len(tokens))

        # A document needs block_size inputs plus one target.
        lengths = stops - starts
        valid = lengths > self.block_size

        self.starts = starts[valid]
        self.stops = stops[valid]

        if len(self.starts) == 0:
            raise ValueError(
                "No documents are long enough for this block size"
            )

        # Sampling proportional to valid window counts avoids
        # oversampling very short documents.
        self.windows = (
            self.stops - self.starts - self.block_size
        ).astype(np.int64)

        self.cumulative = np.cumsum(self.windows)
        self.total_windows = int(self.cumulative[-1])

        print({
            "documents": len(starts),
            "usable_documents": len(self.starts),
            "usable_windows": self.total_windows,
            "block_size": self.block_size,
        })

    def sample(self, batch_size, device="cpu"):
        draws = self.rng.integers(
            0, self.total_windows, size=batch_size
        )

        doc_indices = np.searchsorted(
            self.cumulative, draws, side="right"
        )

        previous = np.where(
            doc_indices == 0,
            0,
            self.cumulative[np.maximum(doc_indices - 1, 0)]
        )

        positions = self.starts[doc_indices] + draws - previous

        x = np.stack([
            np.asarray(
                self.tokens[p:p + self.block_size],
                dtype=np.int64
            )
            for p in positions
        ])

        y = np.stack([
            np.asarray(
                self.tokens[p + 1:p + self.block_size + 1],
                dtype=np.int64
            )
            for p in positions
        ])

        return (
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device)
        )
