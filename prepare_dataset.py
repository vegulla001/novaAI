"""Prepare a novel corpus with Gutenberg cleanup, deduplication and book-level split."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import numpy as np
from tokenizer import BPETokenizer
from genres import load_genres, genre_prefix

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
    p.add_argument("--genres-file", default="dataset/genres.json")
    p.add_argument("--chunk-chars", type=int, default=0, help="0 keeps books intact (recommended); >0 repeats genre tags between chunks")
    args = p.parse_args()
    if args.chunk_chars != 0 and args.chunk_chars < 200: p.error("--chunk-chars must be 0 or >= 200")
    genre_file = Path(args.genres_file)
    if not genre_file.exists():
        print('No genre metadata found; creating heuristic genre suggestions.')
        from auto_genres import predict_file
        names = sorted(Path(args.input_dir).rglob('*.txt'))
        genre_map = {f.name: predict_file(f)[0] for f in names}
        genre_file.parent.mkdir(parents=True, exist_ok=True)
        genre_file.write_text(json.dumps(genre_map, indent=2), encoding='utf-8')
        print('WARNING: Genre labels are keyword-based guesses. Review', genre_file)
    else:
        genre_map = load_genres(genre_file)
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
        if file.name not in genre_map: raise ValueError(f"Add genres for {file.name} to {args.genres_file}")
        books.append((str(file), text, genre_map[file.name]))
    if len(books) < 2:
        raise ValueError("Add at least two distinct novels to dataset/novels/ for book-level validation.")
    random.Random(args.seed).shuffle(books)
    val_count = max(1, round(len(books) * args.val_ratio))
    val_books = books[:val_count]
    train_books = books[val_count:]
    if not train_books: raise ValueError("No training books left after split.")
    def tagged_corpus(items):
        chunks = []
        for _, content, labels in items:
            prefix = genre_prefix(labels)
            if args.chunk_chars == 0:
                chunks.append(prefix + content)
            else:
                for offset in range(0, len(content), args.chunk_chars):
                    chunks.append(prefix + content[offset:offset + args.chunk_chars])
        return "\n\n".join(chunks)
    train_text = tagged_corpus(train_books)
    val_text = tagged_corpus(val_books)
    # Training on a limited deterministic sample avoids O(vocab_size * corpus_size)
    # BPE training time on huge corpora. Encoding still processes all books.
    # Sample across all training books instead of learning BPE only from the first book.
    sample_chunks = []
    budget = args.tokenizer_chars
    per_book = max(1, budget // len(train_books))
    for _, content, labels in train_books:
        sample_chunks.append(genre_prefix(labels) + content[:per_book])
    sample = "\n".join(sample_chunks)[:budget]
    tokenizer = BPETokenizer.train(sample, vocab_size=args.vocab_size)
    tokenizer.save(out / "tokenizer.json")
    train_ids = np.asarray(tokenizer.encode(train_text), dtype=np.uint32)
    val_ids = np.asarray(tokenizer.encode(val_text), dtype=np.uint32)
    np.save(out / "train.npy", train_ids)
    np.save(out / "val.npy", val_ids)
    metadata = {
        "train_books": [name for name, _, _ in train_books],
        "val_books": [name for name, _, _ in val_books],
        "vocab_size": tokenizer.vocab_size,
        "genre_conditioned": True,
        "chunk_chars": args.chunk_chars,
        "corpus_sha256": hashlib.sha256((train_text + "\0" + val_text).encode("utf-8")).hexdigest(),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "tokenizer_training_chars": len(sample),
        "note": "Validation is held out by entire book, not by character slice."
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
