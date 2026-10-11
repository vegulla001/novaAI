# StoryAI V2 — custom byte-level BPE tokenizer

**No pretrained weights, external model APIs, or pretrained tokenizer.**
`model.py` is your custom RoPE/RMSNorm/SwiGLU Transformer from V1.

## Setup (PowerShell)
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Test tokenizer
```powershell
python tokenizer.py
```

## Train on the bundled sample (pipeline smoke test)
```powershell
python train.py --steps 100 --batch-size 4 --block-size 32 --n-layer 2 --n-head 4 --n-embd 128 --vocab-size 300 --eval-every 25
```

## Generate
```powershell
python generate.py --prompt "The dragon" --tokens 100
```

## Train on your own larger text corpus
Replace `dataset/stories.txt` with UTF-8 text you have rights to use.
```powershell
python train.py --steps 5000 --batch-size 16 --block-size 128 --n-layer 4 --n-head 4 --n-embd 256 --vocab-size 1024
```

## How BPE works
- Starts with 256 byte tokens, supporting any UTF-8 text.
- Repeatedly merges the most common adjacent token pair.
- Saves merge rules in `checkpoints/tokenizer.json`.
- Uses the same rules for encoding training prompts and generated output.

The BPE trainer is deliberately simple and recomputes pair frequencies at each step;
it is good for learning and small corpora, not optimized for very large datasets.
Vocabulary and model checkpoint must stay together; changing the vocabulary requires retraining.
Training on the tiny bundled sample will mostly memorize repetitive text.
