# StoryAI V8 — Structured story dataset pipeline

This package extends V7 with a **real dataset download + conversion pipeline**. It does **not** contain 1,000 bundled records, does not claim a trained genre classifier, and does not fix the prose quality of the V5 checkpoint. The default HF dataset's schema, access and license must be checked at download time.

## Windows PowerShell

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-structured.txt
python download_structured.py --count 1000
python prepare_structured.py
python prepare_structured_tokens.py
```

`download_structured.py` prints actual field names and saves unmodified records in `dataset/structured/raw.jsonl`. `prepare_structured.py` converts records with a nonempty introduction/story field into `instruction`/`response` examples; fewer than 1,000 may qualify. The conversion never invents characters, genre labels or scene descriptions. Inspect records manually and respect dataset licenses.

`prepare_structured_tokens.py` uses the existing tokenizer by default and writes train/val arrays to `dataset/processed_structured/` (leaves your original V5 processed data intact). To train a separate tokenizer, pass `--new-tokenizer`, but you must train a fresh model rather than resume an incompatible checkpoint.

## Training

Inspect the existing training CLI with `python train.py --help`. If it has a `--data-dir` flag, point it at `dataset/processed_structured`; otherwise change the training script's data paths after backing it up. A 1,000-record dataset is only a smoke test for instruction formatting, **not enough** to train a fluent novel generator from scratch. Mix a much larger licensed prose corpus with properly annotated examples, use source-level validation splits, and evaluate character adherence separately.

## Auto genres + foundation (from V7)

```powershell
python auto_genres.py --input-dir dataset/novels --output dataset/genres_auto.json
python story_foundation.py --title "The Lost Kingdom" --premise "A young warrior discovers an ancient kingdom" --characters Arin Lyra Malakar --chapters 8 --introduction-style profiles --output output/lost_kingdom_foundation.json
```

Auto genres remain keyword-based suggestions, not a trained classifier. Foundation profiles are planning metadata, not guaranteed model behavior.
