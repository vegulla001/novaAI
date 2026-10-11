# StoryAI V7: Foundation + automatic genre suggestions

This project **does not** contain a trained genre classifier. Automatic genres are transparent **keyword heuristics**, not accurate learned classification. You can override or edit the labels. It also cannot make an undertrained model write fluent novels.

## Create story foundation

```bash
python story_foundation.py --title "The Lost Kingdom" --premise "A young warrior finds an ancient map and fights to save a lost kingdom" --characters Arin Lyra Malakar --chapters 8 --introduction-style profiles --output output/lost_kingdom_foundation.json
```

Edit the JSON: character ages, backstories, goals, world rules, chapter events, and locations. Change `introduction_style` to `natural` to introduce characters inside the narrative, or `prologue` for a short premise-based prologue. The profiles and plans are **structured scaffolding**; the model does not invent those details automatically.

## Generate TXT using your existing V5 checkpoint

Copy your trained `storyai_v5_best.pt` into `checkpoints/` and its **matching** `tokenizer.json` into `dataset/processed/`. Then:

```bash
python generate_novel.py --foundation output/lost_kingdom_foundation.json --chapters 8 --tokens-per-chapter 300 --checkpoint checkpoints/storyai_v5_best.pt --output output/lost_kingdom.txt
```

Use the same model.py/tokenizer.py architecture and vocabulary used for the checkpoint. The script checks tokenizer SHA256 if present.

## Suggest genres for unlabeled books

```bash
python auto_genres.py --input-dir dataset/novels --output dataset/genres_auto.json
```

Inspect `dataset/genres_auto.json` and copy it to `dataset/genres.json` if suitable. The existing `prepare_dataset.py` generates heuristic labels only when `dataset/genres.json` does not exist; otherwise it uses your labels. Avoid overwriting previously reviewed metadata.

```bash
python prepare_dataset.py --vocab-size 1024 --min-chars 1000
```

**Do not reprepare the tokenizer if your goal is to use your existing trained checkpoint.** Changing token IDs breaks checkpoint compatibility. Use fresh data preparation only for a new training run.

## Model-quality warning

Your last 20.98M parameter model trained on ~1.18M tokens and overfit (validation loss worsened after step 400). This release improves organization, not the model's English fluency. Reliable learned multi-label genre prediction requires many correctly labeled books and a trained classifier; keyword matches are only rough suggestions.
