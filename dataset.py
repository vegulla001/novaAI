"""Memory-map pretokenized uint32 arrays; only copy sampled batches to GPU."""
import numpy as np
import torch

def load_tokens(path):
    return np.load(path, mmap_mode="r")

def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size:
        raise ValueError(f"Need > {block_size} tokens; found {len(data)}")
    starts = np.random.randint(0, len(data) - block_size, size=batch_size)
    x = np.stack([data[i:i+block_size] for i in starts]).astype(np.int64)
    y = np.stack([data[i+1:i+block_size+1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
