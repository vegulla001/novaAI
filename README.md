# NovaAI V5 — genre-conditioned, multi-chapter novel drafting

This project trains **your own randomly initialized PyTorch Transformer**. No pretrained models, hosted AI APIs, or TorchText are required.

## Important limitations

- Genre *conditioning* is implemented: a novel can have several labels (ACTION + ADVENTURE, etc.). Automatic genre **classification** is **not trained**; four books are insufficient for a useful classifier.
- `generate_novel.py` produces a complete **multi-chapter draft file** in Markdown, but it cannot promise coherent, publication-quality novel-length writing. With four novels and a short context window, repetition and contradictions are likely.
- V5 re-trains on genre-tagged text. **Do not resume a V4 checkpoint against this new corpus**; the tokenized data and tokenizer may differ. Back up your existing checkpoint and processed data before running preparation.

## 1. Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Prepare novels and genre labels

Put at least **two distinct UTF-8 novels** in `dataset/novels/`. Two short demonstration texts are included for a smoke test; replace them with your real novels.

Edit `dataset/genres.json` to assign one or more genres to **every `.txt` filename**:

```json
{
  "sherlock_holmes.txt": ["MYSTERY", "ADVENTURE"],
  "dracula.txt": ["HORROR", "SUPERNATURAL"],
  "frankenstein.txt": ["SCI_FI", "HORROR"],
  "the_time_machine.txt": ["SCI_FI", "ADVENTURE"]
}
```

Genres: ACTION, ADVENTURE, FANTASY, ROMANCE, MYSTERY, THRILLER, HORROR, SCI_FI, COMEDY, DRAMA, SLICE_OF_LIFE, SUPERNATURAL, HISTORICAL.

```powershell
python prepare_dataset.py --vocab-size 512 --min-chars 1000
```

The preparation script inserts genre prefixes repeatedly into the training text. Splitting is **by entire book**, so with only four books validation is necessarily noisy. Genre prefixes are ordinary BPE-encoded text, not reserved vocabulary tokens.

## 3. Train on CPU

```powershell
python train.py --steps 2000 --batch-size 4 --grad-accum 4 --block-size 128 --n-layer 4 --n-head 4 --n-embd 128 --dropout 0.1 --eval-every 100 --amp off --patience 10
```

Outputs: `checkpoints/storyai_v5.pt` (latest resume state) and `checkpoints/storyai_v5_best.pt` (best observed validation loss). **V5 checkpoints are separate from V4.** Resume only if the dataset/tokenizer match:

```powershell
python train.py --resume --steps 3000 --checkpoint checkpoints/storyai_v5.pt --batch-size 4 --grad-accum 4 --amp off
```

## 4. Generate a multi-chapter novel draft

```powershell
python generate_novel.py --genres ACTION ADVENTURE --title "The Lost Kingdom" --premise "A young warrior searches for a lost kingdom" --chapters 8 --tokens-per-chapter 500 --output output/lost_kingdom.md
```

Use `--outline-file chapter_outlines.txt` for one outline per line. The output file is written **incrementally** so finished chapters remain on disk if generation is interrupted.

A 50,000-word novel requires vastly more tokens than the defaults and much stronger training/context memory. This script is a draft-generation scaffold, not a high-quality long-form writing solution.

## Notes

- For large datasets, replace the simple Python BPE implementation with a more efficient **self-trained** tokenizer.
- The model's `generate` method recomputes attention at each token; KV caching would substantially improve generation speed.
- Do not delete your V4 checkpoint. Genre conditioning is a new training objective and the old checkpoint cannot safely be resumed as-is.
