from pathlib import Path
import torch
from tokenizer import CharTokenizer

def load_data(path="dataset/stories.txt", split=0.9):
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Dataset too small: add at least 100 characters.")
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(tokens) * split)
    return tokens[:n], tokens[n:], tokenizer

def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size:
        raise ValueError("Not enough tokens for block_size; reduce block_size or add text.")
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
