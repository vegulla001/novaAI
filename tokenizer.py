"""Byte-level BPE tokenizer implemented from scratch (no pretrained vocabulary)."""
import json
from collections import Counter
from pathlib import Path

class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges, start=256):
            if a not in self.vocab or b not in self.vocab:
                raise ValueError("Invalid BPE merge order")
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.vocab_size = len(self.vocab)

    @staticmethod
    def _merge(ids, pair, new_id):
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    @classmethod
    def train(cls, text, vocab_size=512, min_frequency=2):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (one token per byte)")
        ids = list(text.encode("utf-8"))
        merges = []
        while len(merges) + 256 < vocab_size:
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            ids = cls._merge(ids, pair, 256 + len(merges))
            merges.append(pair)
        return cls(merges)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # Apply merges in training order. Every merge is a new token.
        for new_id, pair in enumerate(self.merges, start=256):
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"type": "byte_bpe_v1", "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte_bpe_v1":
            raise ValueError("Expected StoryAI byte BPE tokenizer")
        return cls(data["merges"])

if __name__ == "__main__":
    sample = "A dragon crossed the valley. A dragon returned! 🐉"
    tok = BPETokenizer.train(sample * 30, vocab_size=300)
    assert tok.decode(tok.encode(sample)) == sample
    print("Tokenizer roundtrip passed. Vocabulary:", tok.vocab_size)
