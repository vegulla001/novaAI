# StoryAI from scratch (PyTorch only)

No pretrained models or third-party model architectures are downloaded. All weights start randomly initialized. `model.py` implements a decoder-only Transformer with RoPE, RMSNorm, and SwiGLU.

## Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --steps 1000 --batch-size 8 --block-size 128
python generate.py --prompt "The dragon" --tokens 200
```

The provided stories.txt is only a smoke-test dataset, so generation will be repetitive or incoherent. Replace it with much more original, public-domain, or properly licensed training text. All characters in your generation prompt must exist in the dataset's vocabulary; otherwise tokenizer.encode raises an error. Save your tokenizer alongside each checkpoint.

Files:
- `model.py`: full neural architecture, with `StoryAI` and `ModelConfig` aliases
- `tokenizer.py`: own character tokenizer
- `dataset.py`: 90/10 sequential train/validation split and random next-token batches
- `train.py`: training, validation, gradient clipping, checkpointing
- `generate.py`: checkpoint loading and sampling

For reliable evaluation on real novels, split by document rather than a contiguous token boundary, and use a held-out set of books. For bigger models add checkpoint-resume and mixed precision.
