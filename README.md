# StoryAI V4 — GPU-ready training from scratch

Your own Transformer (`model.py`), your own byte-level BPE tokenizer (`tokenizer.py`),
no pretrained weights and no third-party model APIs.

## Install (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Prepare data
Place **at least two distinct** UTF-8 novels in `dataset/novels/`. Use books you have rights to train on.
```powershell
python prepare_dataset.py --vocab-size 512 --min-chars 1000
```
For a quick smoke test, the archive includes two **synthetic demonstration texts**.

## 2. Train (CPU quick check)
```powershell
python train.py --steps 3 --batch-size 2 --grad-accum 2 --block-size 32 --n-layer 2 --n-head 2 --n-embd 64 --eval-every 1 --eval-batches 2 --amp off
```

## 3. Train on GPU
```powershell
python train.py --steps 5000 --batch-size 8 --grad-accum 8 --block-size 256 --n-layer 6 --n-head 8 --n-embd 512 --lr 3e-4 --min-lr 3e-5 --warmup 200 --eval-every 100 --amp auto
```
Effective batch size is `batch-size × grad-accum`. Choose settings to fit GPU memory.
`--amp auto` uses BF16 when supported, otherwise FP16; CPU uses full precision.

## 4. Resume after interruption
```powershell
python train.py --resume --steps 6000 --batch-size 8 --grad-accum 8 --eval-every 100
```
Resume loads architecture from checkpoint. Keep your tokenizer and processed dataset unchanged.
The new `--steps` is the **total target** optimizer updates, not additional updates.
The learning-rate schedule is recalculated against the new total target on resume.
Only load checkpoints you trust.

## 5. Generate
```powershell
python generate.py --prompt "The dragon" --tokens 200
```
The prompt can contain any UTF-8 characters. A small demo dataset is only for pipeline testing.

## Features
- GPU CUDA autocast (BF16/FP16) and FP16 gradient scaling
- Gradient accumulation and gradient clipping
- Learning-rate warmup + cosine decay
- Book-level held-out validation
- Atomic checkpoint writes and resume (model, optimizer, RNG, scaler)
- Tokenizer and dataset metadata SHA-256 checks before resuming

## Notes
- Resuming with a changed total step target alters the future cosine schedule.
- This is single-process training (not multi-GPU DDP).
- Dataset preprocessing and BPE are educational implementations, not yet optimized for massive corpora.
- Checkpoints and training data are excluded from Git by `.gitignore`.
