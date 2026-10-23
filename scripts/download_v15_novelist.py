
from datasets import load_dataset
from pathlib import Path
import json

out = Path("dataset/v15/novelist_raw")
out.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "Dxniz/Novelist",
    split="train",
    streaming=True,
)

counts = {}

with (out / "english_narrative.jsonl").open(
    "w", encoding="utf-8"
) as f:
    for row in dataset:
        kind = row.get("record_type", "")
        counts[kind] = counts.get(kind, 0) + 1

        if row.get("language_code") != "en":
            continue

        if kind not in ("story_scene", "reasoned_story_scene"):
            continue

        if row.get("quality_accepted") is False:
            continue

        prose = (
            row.get("answer_text")
            or row.get("text")
            or ""
        ).strip()

        if len(prose.split()) < 200:
            continue

        f.write(json.dumps({
            "story_id": row.get("story_id"),
            "book_id": row.get("book_id"),
            "genre": row.get("genre"),
            "record_type": kind,
            "text": prose,
        }, ensure_ascii=False) + "\n")

print("Source record types:", counts)
print("Saved to:", out)
