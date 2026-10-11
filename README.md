# StoryAI V3 — Train from a folder of novels

Your model uses PyTorch, RoPE, RMSNorm, SwiGLU, and **randomly initialized weights**. The tokenizer is custom byte-level BPE, not pretrained.

## 1. Install (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Add novels

Download **Plain Text UTF-8** books you have rights to use. Place **at least two different** `.txt` novels in `dataset/novels/`, for example `sherlock.txt`, `dracula.txt`. For meaningful validation, use many books. The example `dataset/stories.txt` from V2 is not used in V3.

## 3. Prepare once

```powershell
python prepare_dataset.py --vocab-size 512 --min-chars 1000
```

This script cleans Gutenberg start/end wrappers where present, skips identical books, splits **by book** (not by passage), learns BPE merges from training text only, and writes `train.npy`, `val.npy`, `tokenizer.json`, and `metadata.json` to `dataset/processed/`.

The BPE trainer is an educational pure-Python implementation; its training sample defaults to the first 150,000 characters of training text for speed. For large-scale use, replace it with an efficient tokenizer implementation that you build yourself.

## 4. Train

```powershell
python train.py --steps 1000 --batch-size 8 --block-size 128 --n-layer 4 --n-head 4 --n-embd 128
```

For a small CPU smoke test, use `--steps 10 --batch-size 2 --block-size 32 --n-layer 2 --n-head 2 --n-embd 64 --eval-every 5`.

## 5. Generate

```powershell
python generate.py --prompt "The " --tokens 100
```

The trained checkpoint is `checkpoints/storyai.pt`; keep it with `dataset/processed/tokenizer.json`. Both are ignored by Git. This version does not implement checkpoint resume, mixed precision, or gradient accumulation yet.

## Caveats

- Gutenberg boilerplate cleanup depends on standard START/END markers; inspect unusual books manually.
- Deduplication catches normalized **exact** copies, not different editions or near duplicates.
- UTF-8 byte BPE can decode incomplete byte sequences during generation with replacement characters; this is expected for a byte-level educational tokenizer.
- Concatenated books can create training windows crossing book boundaries. To avoid this in a production corpus, build book-aware sampling.
- `train.npy` and `val.npy` are memory-mapped for sampling; preparing them still loads and encodes each split in memory.
- Only train on books you have permission to use; public-domain status depends on jurisdiction.
