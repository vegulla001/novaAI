#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m training.v13.gpu_preflight --require-gpu --data-dir dataset/processed_structured --block-size "${BLOCK_SIZE:-2048}"
mkdir -p logs checkpoints
python -m training.v13.train_long_context \
  --checkpoint checkpoints/storyai_v8_best.pt \
  --data-dir dataset/processed_structured \
  --tokenizer dataset/processed_structured/tokenizer.json \
  --output checkpoints/storyai_v13_long_context.pt \
  --block-size "${BLOCK_SIZE:-2048}" \
  --steps "${STEPS:-1000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-16}" \
  --eval-every 100 2>&1 | tee logs/v13_gpu.log
