#!/usr/bin/env bash
set -euo pipefail
python -m training.v15.prepare --corpus "${CORPUS:-dataset/v12/prose_processed}" --vocab-size "${VOCAB_SIZE:-8192}" --min-train-tokens "${MIN_TOKENS:-5000000}"
python -m training.v15.train_lm --block-size "${BLOCK_SIZE:-2048}" --steps "${LM_STEPS:-10000}" --batch-size "${BATCH_SIZE:-1}" --grad-accum "${GRAD_ACCUM:-16}" --min-train-tokens "${MIN_TOKENS:-5000000}"
python -m training.v15.train_instruction --train "${INSTRUCTION_TRAIN:-dataset/v12/pilot_expanded/train.jsonl}" --val "${INSTRUCTION_VAL:-dataset/v12/pilot_expanded/val.jsonl}" --min-pairs "${MIN_INSTRUCTION_PAIRS:-1000}"
