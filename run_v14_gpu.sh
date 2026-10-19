#!/usr/bin/env bash
set -euo pipefail
python -m training.v13.gpu_preflight --require-gpu --data-dir dataset/processed --block-size "${BLOCK_SIZE:-2048}"
python -m training.v13.train_long_context --checkpoint checkpoints/storyai_v8_best.pt --tokenizer dataset/processed/tokenizer.json --data-dir dataset/processed --block-size "${BLOCK_SIZE:-2048}" --steps "${LM_STEPS:-1000}" --batch-size "${BATCH_SIZE:-1}" --grad-accum "${GRAD_ACCUM:-16}"
python -m training.v14.train_instructions --steps "${INSTRUCTION_STEPS:-1500}"
echo 'Training stages complete. Generate: python -m story_engine.v14.novel --spec story_spec.example.json'
