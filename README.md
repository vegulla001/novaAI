# StoryAI V6 — Improving Readability (experimental)

This is an original randomly initialized PyTorch Transformer (RoPE, RMSNorm, SwiGLU), not a pretrained model. It does **not** guarantee fluent English or a coherent novel. Four novels are far too few to train a strong novelist from scratch.

## Changes from V5
- Genre labels preserved, but genre markers are placed once per book by default instead of every 800 characters. This reduces control-token leakage and artificial mid-sentence interruptions.
- BPE vocabulary sample is spread across training books rather than drawn entirely from the beginning of the first book.
- TXT-only novel output, configurable chapter outline, repetition penalty, and removal of visible genre tags.
- New V6 checkpoint names avoid overwriting V5 checkpoints.
- Existing V5 validation-loss monitoring, early stopping and best-checkpoint saving retained.

## Windows PowerShell

```powershell
python -m pip install -r requirements.txt
```

Put your own UTF-8 public-domain novels in `dataset/novels/`, and label **each file** in `dataset/genres.json`. At least two distinct books are needed. Remove demo books or label them if you keep them.

```powershell
python prepare_dataset.py --vocab-size 1024 --min-chars 1000 --chunk-chars 0
python train.py --steps 2000 --batch-size 4 --grad-accum 4 --block-size 128 --n-layer 4 --n-head 4 --n-embd 128 --dropout 0.1 --eval-every 100 --amp off --patience 10
python generate_novel.py --genres ACTION ADVENTURE --title "The Lost Kingdom" --premise "A warrior searches for a lost kingdom" --chapters 3 --tokens-per-chapter 300 --outline-file chapter_outline.txt --output output/lost_kingdom.txt
```

Best weights: `checkpoints/storyai_v6_best.pt`. Latest resumable state: `checkpoints/storyai_v6.pt`.

**Do not use `--resume` with V4/V5 checkpoints after regenerating the tokenizer.** The token-ID mapping changes. Keep your old checkpoints and tokenizer backed up. Increasing context or model size also requires a new run or an explicitly designed migration.

## What to expect

Training 2,000 steps on four books will likely produce misspellings and weak coherence. Better generation needs much more clean text, an appropriate vocabulary, and enough compute. A 128-token context cannot remember an entire chapter or novel. The outline is a prompt, not a reliable long-term memory system.

For an honest quality check, generate 100–200 tokens every few hundred steps and compare outputs. Monitor held-out validation loss. A good-looking text file is not proof that the model can write a novel.
