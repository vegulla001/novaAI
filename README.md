# Story AI v16 — guarded foundation recovery

This fixes *how* foundation pretraining is resumed and checked. It is **not** a claim that the current 31.47M checkpoint can write coherent novels or reason like a large model. No additional model weights are included.

## Install
Extract this archive into your existing `story-ai` project, preserving `training/v16/foundation_recovery.py`. It does not overwrite `training/v15/train.py`, model weights, or datasets.

## Step 1 — Local Windows CPU preflight (NO TRAINING)
```powershell
python -m training.v16.foundation_recovery `
  --checkpoint checkpoints/pretrain_2000.pt `
  --data-dir dataset/v15_prose_expanded `
  --report reports/v16_foundation_recovery_preflight.json
```

It verifies the checkpoint format and nonzero step, tokenizer SHA-256 against the checkpoint and dataset report, token ID ranges, and 1D arrays. It checks a *sample* of train/validation windows for identical overlap (not a complete contamination test). It prints the exact training command and stops; it does NOT start training.

## Step 2 — GPU (only if you decide to invest in foundation training)
First upload `checkpoints/pretrain_2000.pt`, `dataset/v15_prose_expanded/{tokenizer.json,build_report.json,train.npy,val.npy}`, `training/v15/model.py`, `training/v15/train.py`, and `training/v16/foundation_recovery.py`. These are inputs, not included in the ZIP.

```bash
cd /workspace/story-ai
PYTHONPATH=/workspace/story-ai python -u -m training.v16.foundation_recovery \
  --checkpoint checkpoints/pretrain_2000.pt \
  --data-dir dataset/v15_prose_expanded \
  --out-dir checkpoints/v16_foundation_recovery_01 \
  --report reports/v16_foundation_recovery_preflight.json \
  --steps 600 --batch-size 4 --grad-accum 8 \
  --lr 8e-5 --min-lr 8e-6 --warmup 40 \
  --eval-every 50 --save-every 50 --eval-batches 20 \
  --device cuda --execute
```

This performs **600 new updates**, 19,660,800 token positions (~0.82 corpus equivalents), using the existing step-2000 pretrained weights with a fresh optimizer/schedule. The initial output `BASELINE_FIXED` is the comparable baseline for this run. It saves `pretrain_best.pt` and `pretrain_latest.pt` in a NEW checkpoint directory. Your original and other v16 checkpoints are unchanged.

**Important:** `--restart-from` is intentional. `--resume` would reuse optimizer state and apply the new schedule at absolute step 2000, risking an LR jump or floor. Fresh optimizer is a tradeoff, not inherently superior in every training scenario.

## How to decide whether to continue
- If validation loss is not improving meaningfully across checkpoints, stop rather than pay for more steps. A low or decreasing loss alone is insufficient.
- Compare unrestricted generation using the identical prompt, tokenizer, and decoding settings on `checkpoints/pretrain_2000.pt` and the best recovery checkpoint.
- The corpus has only heuristic deduplication; monitor repetition and contamination. Do not assume 600 updates will provide general reasoning ability.
- Do not run another character-specific SFT until language foundation and context adherence improve independently.

## Tests
```powershell
python -m unittest discover -s tests -p test_foundation_recovery.py -v
```
