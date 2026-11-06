# Story AI v16 — reasoning repair, CPU-first

This package **does not claim to fix model weights**. It fixes the next training-data / evaluation stage after all three v16 models scored 6/12 and 0/6 counterfactual pairs. It produces balanced synthetic examples with verifiable answers and explanations, and evaluates the model's *actual* conditional probabilities on held-out names and objects. There is no rule-based inference fallback, no generated "approved" novel text, and no GPU run triggered automatically.

## Install

Extract the ZIP into the **root** of `story-ai`. It adds only:
- `training/v16/reasoning_curriculum.py`
- `training/v16/evaluate_reasoning_repair.py`
- `tests/test_reasoning_curriculum.py`

It does not replace `training/v15/train.py`, `training/v15/model.py` or existing checkpoints.

## Step 1 — local Windows CPU: structural tests

```powershell
cd C:\Users\vegul\OneDrive\Desktop\story-ai
python -m unittest discover -s tests -p test_reasoning_curriculum.py -v
```

## Step 2 — local Windows CPU: create and validate curriculum

The **foundation** checkpoint must be pretrain-mode; use the recovered foundation if locally backed up, otherwise `checkpoints/pretrain_2000.pt`. This example uses the recovered one.

```powershell
python -m training.v16.reasoning_curriculum `
  --checkpoint checkpoints/v16_foundation_recovery_01/pretrain_best.pt `
  --tokenizer dataset/v15_prose_expanded/tokenizer.json `
  --out-dir dataset/v16_reasoning_repair_01
```

Creates 10,800 training samples (5,400 contrast pairs) and 1,200 validation samples (600 contrast pairs). Labels and causal explanations are constructed from scene facts, not model output. Test names, objects and some settings are disjoint from training. **It does not train a model.** Validation examples are synthetic and still have limited template variety, so broad generalization is not guaranteed. The baseline's original 12 scenes are not used verbatim.

## Step 3 — local CPU: baseline reasoning comparison

Run a SMALL subset to avoid expensive 31M CPU inference. Use `--pairs-per-skill 4` = 48 evaluations per checkpoint. On GPU (if available later), raise to 50 or 100.

```powershell
python -m training.v16.evaluate_reasoning_repair `
  --checkpoint checkpoints/pretrain_2000.pt `
  --checkpoint checkpoints/v16_foundation_recovery_01/pretrain_best.pt `
  --checkpoint checkpoints/v16_novel_writer_01/sft_best.pt `
  --device cpu `
  --pairs-per-skill 4 `
  --out reports/v16_reasoning_repair_baseline.json
```

Look at **paired** accuracy in addition to 2-choice accuracy. A model that always answers the same option can reach 50% and still score 0 paired. The test measures **forced-choice ranking**, not free generation, explanations, or genuine internal reasoning. Exact tokenization and candidate length can affect scores.

## Optional future GPU stage — DO NOT START without reviewing baseline and dataset

The existing `training.v15.train` script supports response-only supervised training from a pretrain-mode checkpoint. It can read the generated `train_sft.pt` and `val_sft.pt` files. It *must* run from a new checkpoint directory to preserve the foundation. The dataset and source code will need to be uploaded first.

Example only (not automatically executed):

```bash
cd /workspace/story-ai
PYTHONPATH=/workspace/story-ai python -u -m training.v15.train \
  --mode sft \
  --data-dir dataset/v15_prose_expanded \
  --sft-dir dataset/v16_reasoning_repair_01 \
  --init-from checkpoints/v16_foundation_recovery_01/pretrain_best.pt \
  --out-dir checkpoints/v16_reasoning_repair_01 \
  --max-steps 200 --batch-size 4 --grad-accum 8 \
  --lr 2e-5 --min-lr 2e-6 --warmup-steps 15 \
  --eval-every 25 --save-every 25 --eval-batches 20 \
  --device cuda
```

**Warning:** Pure reasoning SFT can worsen prose ability. A real writing-and-reasoning model will ultimately need a measured mixture of genuine coherent prose, story-state tasks and reasoned plans, plus larger-scale generalization tests. **Do not take lower SFT loss as proof of improved reasoning.** If the benchmark remains near 50% / 0 paired, stop and inspect data/objective rather than escalating steps. Before publishing any model, independently test novel scene continuation and original free-running reasoning.

## Why this is not a promised 'reasoning fix'

The model previously picked a constant preferred answer on each pair. This provides a falsifiable training target for that failure. It is insufficient evidence of long-horizon planning, internal thinking, general comprehension or autonomous novel writing. A 31M-param model may need substantially more quality training data and compute to achieve the full goal.
