# StoryAI V13: real context length upgrade

Your V8 checkpoint was trained at **512 tokens**. This patch increases the **context window** to **1024** (or **2048**) using the existing RoPE architecture. The token vocabulary stays the same; do **not** change the tokenizer. Context length is different from maximum generated output length.

The patch is an **independent continuation experiment**, not a claim that novel quality is fixed. No weights are downloaded. V8 and V12 checkpoints are not overwritten.

## Install

Copy `training/v13/train_long_context.py` into the same path in your project. Keep your original `model.py` unchanged.

## Train with longer context (PowerShell, from project root)

```powershell
python -m training.v13.train_long_context --block-size 1024 --steps 1000 --batch-size 2 --grad-accum 8
```

For a GPU with sufficient VRAM, try `--block-size 2048 --batch-size 1 --grad-accum 16`. A 2048-token sequence needs much more memory/time than 512. On CPU, test with `--steps 2 --batch-size 1 --grad-accum 1` first.

Input: `checkpoints/storyai_v8_best.pt` and `dataset/processed/{train.npy,val.npy,tokenizer.json}`.
Output: `checkpoints/storyai_v13_long_context.pt`, `output/v13_long_context/report.json`.

**Critical:** RoPE allows extending the context *structurally* without changing weight tensor sizes, but useful long-range attention must be learned with continued training. Merely setting block_size=2048 does not give 2048-token comprehension.

This uses the original token streams, not the 16 authored scenes. Training data quality, model capacity (7.47M parameters), and instruction-following remain separate problems. For chapter generation, plan scenes and carry forward a compact story bible; a 2048-token context is not an entire novel.

If `dataset/processed/train.npy` is missing, restore your V8 training token arrays; don't substitute token arrays made using a different tokenizer.
