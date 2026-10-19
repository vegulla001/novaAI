# StoryAI V15 — Fresh training + existing V10 NLU

**Status:** Integrated source code and safety gates; NOT a trained, validated novel-writing product. This is a fresh model, not a V14 checkpoint fix.

## Why training will NOT start immediately with the bundled datasets

- The original V14 1,024-vocabulary checkpoints cannot be loaded into a fresh 8,192-vocabulary model.
- The current 16 reviewed authored instruction-training examples are insufficient; V15 **refuses** SFT by default until at least 1,000 examples are provided. The auto-extracted candidates are NOT verified.
- The fiction corpus must contain at least 5,000,000 **new-tokenizer** training tokens, with document-disjoint validation. The preprocessor refuses insufficient data.
- These gates avoid another costly GPU run that cannot achieve the requested outcome. Do not lower them merely to bypass the checks.
- The original V10 NLU is included under `nlu/`; to use natural-language story-editing commands you need `checkpoints/storyai_v10_nlu.pt`, or train it using `python -m nlu.train` with `dataset/nlu`.

## Setup

```bash
pip install -r requirements-v15.txt
python -m training.v15.prepare --corpus dataset/v12/prose_processed --vocab-size 8192
python -m training.v15.train_lm --steps 10000 --block-size 2048 --batch-size 1 --grad-accum 16
python -m training.v15.train_instruction --train /path/to/REVIEWED_train.jsonl --val /path/to/REVIEWED_val.jsonl
python -m story_engine.v15.novel --spec story_spec.example.json --commands story_commands.example.txt --outdir output/v15_novel
```

Or use `bash run_v15_gpu.sh` once the corpus and reviewed instruction pairs are ready. Use `--layers`, `--heads`, `--embedding`, `--block-size` to size the GPT to available VRAM. The default is 8 layers, 8 heads, 512 embedding, 2048 context. A 4096 context can use much more VRAM. **No pretrained weights or external AI inference APIs**.

## NLU integration

The optional `--commands` file contains one user instruction per line. The trained V10 NLU interprets these as edits to the structured story spec (character, role, relationship, plot, chapter title). Unsupported or low-confidence commands are **rejected** and recorded in `nlu_report.json`, never silently applied. NLU is NOT a substitute for prose-model instruction-following. If you omit `--commands`, no NLU checkpoint is needed.

## Data formats

Instruction JSONL examples must include `characters`, `setting`, `obstacle`, `outcome`, `story`, and either `goal` or `objective`. Use independently reviewed fiction and hold out entire source families. Existing `dataset/v12/pilot_expanded` is for tests only, not sufficient for training.

## Output

`novel.md`, `chapter_XX.md`, `story_state.json`, `outline.json`, `nlu_report.json`, `quality_report.json`. Rejected scenes are **explicitly marked**, never presented as successful prose. The current engine records story state but does not yet enforce semantic continuity or plan unique chapter arcs; those require additional model and evaluation work.

## Limitations

This code has not been GPU-trained or validated end-to-end with an adequate corpus and 1,000+ reviewed pairs. A low loss or passing basic gate does not guarantee coherent fiction. The current naive scene generation repeats a short training-aligned prompt and will not by itself deliver true novel continuity. Do not rent another GPU until the dataset gates pass and you have enough data.
