import argparse
import json
import time
from pathlib import Path

import numpy as np
from tokenizer import BPETokenizer


def serialize(row):
    return (
        "<INSTRUCTION>\n"
        + row["instruction"]
        + "\n<STORY>\n"
        + row["response"]
        + "\n<END_STORY>\n\n"
    )


def load_examples(path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                yield serialize(row)
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid example at {path}:{line_number}: {exc}"
                ) from exc


def tokenize_split(tokenizer, input_file, output_file):
    print(f"\nProcessing: {input_file}", flush=True)

    all_tokens = []
    total_examples = 0
    start = time.time()

    for text in load_examples(input_file):
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

        total_examples += 1

        if total_examples % 10 == 0:
            elapsed = time.time() - start

            print(
                f"Examples: {total_examples} | "
                f"Tokens: {len(all_tokens):,} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )

    ids = np.asarray(all_tokens, dtype=np.uint32)
    np.save(output_file, ids)

    print(
        f"Completed {total_examples} examples, "
        f"{len(ids):,} tokens",
        flush=True,
    )

    return len(ids)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="dataset/structured",
    )

    parser.add_argument(
        "--output-dir",
        default="dataset/processed_structured",
    )

    parser.add_argument(
        "--tokenizer",
        default="dataset/processed/tokenizer.json",
    )

    parser.add_argument(
        "--new-tokenizer",
        action="store_true",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--tokenizer-chars",
        type=int,
        default=150000,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting StoryAI V8 preprocessing...", flush=True)

    if args.new_tokenizer:
        print("Training new BPE tokenizer...", flush=True)

        sample_parts = []
        remaining = args.tokenizer_chars

        for example in load_examples(input_dir / "train.jsonl"):
            if remaining <= 0:
                break

            sample_parts.append(example[:remaining])
            remaining -= len(sample_parts[-1])

        sample = "".join(sample_parts)

        tokenizer = BPETokenizer.train(
            sample,
            vocab_size=args.vocab_size,
        )
    else:
        print("Loading existing tokenizer...", flush=True)

        tokenizer = BPETokenizer.load(args.tokenizer)

    tokenizer.save(output_dir / "tokenizer.json")

    print(
        f"Tokenizer vocabulary: {tokenizer.vocab_size}",
        flush=True,
    )

    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "structured": True,
    }

    for split in ("train", "val"):
        metadata[f"{split}_tokens"] = tokenize_split(
            tokenizer,
            input_dir / f"{split}.jsonl",
            output_dir / f"{split}.npy",
        )

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nPreprocessing completed!", flush=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()