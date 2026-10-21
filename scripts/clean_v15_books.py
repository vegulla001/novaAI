import re
import argparse
from pathlib import Path

START = re.compile(
    r"(?im)^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}\s*$"
)

END = re.compile(
    r"(?im)^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}\s*$"
)

def clean_book(text):
    start = START.search(text)
    end = END.search(text)

    if not start or not end or start.end() >= end.start():
        return None

    text = text[start.end():end.start()]

    # Remove standalone illustration placeholders.
    text = re.sub(
        r"(?im)^\s*\[Illustration(?::[^\]\n]*)?\]\s*$",
        "",
        text,
    )

    # Normalize line endings and excessive blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="dataset/v15/sources/books",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/v15/sources/cleaned",
    )
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for path in sorted(source.glob("pg_*.txt")):
        text = path.read_text(encoding="utf-8-sig")
        cleaned = clean_book(text)

        if cleaned is None:
            print(f"REVIEW REQUIRED: {path.name} - markers missing")
            continue

        destination = output / path.name
        destination.write_text(cleaned, encoding="utf-8")

        print(
            f"CLEANED: {path.name} "
            f"({len(text):,} -> {len(cleaned):,} characters)"
        )


if __name__ == "__main__":
    main()