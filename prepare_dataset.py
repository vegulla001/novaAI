"""Prepare a novel corpus with Gutenberg cleanup, deduplication and book-level split."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np
from tokenizer import BPETokenizer

START = re.compile(r"^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b.*$", re.I | re.M)
END = re.compile(r"^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b.*$", re.I | re.M)

def clean(text):
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    start = START.search(text)
    end = END.search(text)
    if start and end and end.start() > start.end():
        text = text[start.end():end.start()]
    elif start:
        text = text[start.end():]
    elif end:
        text = text[:end.start()]
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="dataset/novels")
    p.add_argument("--output-dir", default="dataset/processed")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--min-chars", type=int, default=1000)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer-chars", type=int, default=150000, help="Cap BPE training sample to keep pure-Python training manageable")
    args = p.parse_args()
    if not 0 < args.val_ratio < 1: raise ValueError("--val-ratio must be between 0 and 1")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input_dir).rglob("*.txt"))
    books, seen = [], set()
    for file in files:
        text = clean(file.read_text(encoding="utf-8-sig", errors="replace"))
        if len(text) < args.min_chars:
            print(f"SKIP too short: {file}")
            continue
        digest = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()
        if digest in seen:
            print(f"SKIP duplicate: {file}")
            continue
        seen.add(digest)
        books.append((str(file), text))
    if len(books) < 2:
        raise ValueError("Add at least two distinct novels to dataset/novels/ for book-level validation.")
    random.Random(args.seed).shuffle(books)
    val_count = max(1, round(len(books) * args.val_ratio))
    val_books = books[:val_count]
    train_books = books[val_count:]
    if not train_books: raise ValueError("No training books left after split.")
    train_text = "\n\n".join(text for _, text in train_books)
    val_text = "\n\n".join(text for _, text in val_books)
    # Training on a limited deterministic sample avoids O(vocab_size * corpus_size)
    # BPE training time on huge corpora. Encoding still processes all books.
    sample = train_text[:args.tokenizer_chars]
    tokenizer = BPETokenizer.train(sample, vocab_size=args.vocab_size)
    tokenizer.save(out / "tokenizer.json")
    train_ids = np.asarray(tokenizer.encode(train_text), dtype=np.uint32)
    val_ids = np.asarray(tokenizer.encode(val_text), dtype=np.uint32)
    np.save(out / "train.npy", train_ids)
    np.save(out / "val.npy", val_ids)
    metadata = {
        "train_books": [name for name, _ in train_books],
        "val_books": [name for name, _ in val_books],
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "tokenizer_training_chars": len(sample),
        "note": "Validation is held out by entire book, not by character slice."
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
