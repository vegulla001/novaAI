import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
import truststore

truststore.inject_into_ssl()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="dataset/v15/sources/book_candidates_review.csv",
    )
    parser.add_argument(
        "--outdir",
        default="dataset/v15/sources/books",
    )
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.candidates, encoding="utf-8-sig", newline="") as f:
        candidates = list(csv.DictReader(f))

    results = []
    downloaded = 0
    attempted = 0

    for row in candidates:
        if attempted >= args.limit:
            break

        book_id = row["book_id"].strip()
        title = row["title"].strip()

        if not book_id.startswith("pg_"):
            continue

        gutenberg_id = book_id.removeprefix("pg_")

        # Known duplicate title from the current corpus.
        if gutenberg_id == "15":
            results.append({
                "book_id": book_id,
                "status": "SKIPPED_KNOWN_DUPLICATE",
            })
            continue

        destination = outdir / f"{book_id}.txt"

        if destination.exists():
            results.append({
                "book_id": book_id,
                "status": "ALREADY_DOWNLOADED",
            })
            continue

        # Prefer the official Project Gutenberg download service.
        url = (
            f"https://www.gutenberg.org/ebooks/"
            f"{gutenberg_id}.txt.utf-8"
        )
        attempted += 1
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "StoryAI-CorpusResearch/1.0"
                },
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()

            text = content.decode("utf-8-sig")

            if len(text) < 5000:
                raise ValueError("Downloaded text unexpectedly short")

            if "project gutenberg" not in text.lower()[:10000]:
                raise ValueError("Unexpected text format")

            destination.write_text(text, encoding="utf-8")

            results.append({
                "book_id": book_id,
                "title": title,
                "status": "DOWNLOADED_UNREVIEWED",
                "source_url": url,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })

            downloaded += 1
            print(f"Downloaded {book_id}: {title}")

            time.sleep(2)

        except (
            urllib.error.URLError,
            UnicodeError,
            ValueError,
            TimeoutError,
        ) as exc:
            results.append({
                "book_id": book_id,
                "title": title,
                "status": "ERROR",
                "error": str(exc),
            })
            print(f"Failed {book_id}: {exc}")

    report = outdir.parent / "download_report.json"
    report.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDownloaded: {downloaded}")
    print(f"Report: {report}")

if __name__ == "__main__":
    main()