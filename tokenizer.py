import json
from pathlib import Path

class CharTokenizer:
    """A character-level tokenizer trained only on your own text."""
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(text)

    def encode(self, text):
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters: {sorted(unknown)!r}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
