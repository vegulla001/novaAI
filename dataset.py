from pathlib import Path
import torch
from tokenizer import BPETokenizer

def load_data(path="dataset/stories.txt", vocab_size=512):
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Dataset too small.")
    boundary = int(len(text) * 0.9)
    train_text, val_text = text[:boundary], text[boundary:]
    # Learn merges from training text only to avoid vocabulary leakage.
    tokenizer = BPETokenizer.train(train_text, vocab_size=vocab_size)
    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return train_ids, val_ids, tokenizer

def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size:
        raise ValueError(f"Need > {block_size} tokens in this split, got {len(data)}. Reduce --block-size or add data.")
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
