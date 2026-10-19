# StoryAI V14 — integrated train-and-generate codebase

This is **software-complete for a basic training/generation workflow**, not a pretrained or proven high-quality novelist. It includes existing V8/V12/V13 code, RoPE GPT, a longer-context continuation trainer, instruction fine-tuning on *reviewed* scenes, and a chapter generator that saves an outline, cast, recent events, chapters, and warnings. It uses no external AI service and no pretrained third-party weights.

## Critical missing inputs

The supplied source archive intentionally excluded the user's V8 checkpoint and original `dataset/processed` files. **You must provide:**

- `checkpoints/storyai_v8_best.pt`
- `dataset/processed/tokenizer.json` **matching the V8 checkpoint**
- `dataset/processed/train.npy` and `dataset/processed/val.npy` encoded with that tokenizer

Do **not** silently substitute `dataset/processed_structured/tokenizer.json` or `.npy` unless their tokenizer and checkpoint compatibility is independently verified. The bundled structured files may belong to a different training setup.

## GPU machine setup

Use an NVIDIA CUDA-compatible PyTorch container. An RTX 4090 24 GB is a reasonable initial test platform; larger context and model sizes can require A100/H100 VRAM. Persistent volume recommended. After uploading this folder and the three missing input types:

```bash
python -m pip install -r requirements-gpu.txt
python -m training.v13.gpu_preflight --require-gpu --data-dir dataset/processed --block-size 2048
```

**Smoke test before renting for hours:**

```bash
BLOCK_SIZE=1024 LM_STEPS=2 INSTRUCTION_STEPS=2 GRAD_ACCUM=1 bash run_v14_gpu.sh
```

**Training:**

```bash
BLOCK_SIZE=2048 LM_STEPS=1000 INSTRUCTION_STEPS=1500 BATCH_SIZE=1 GRAD_ACCUM=16 bash run_v14_gpu.sh
```

Train outputs are `checkpoints/storyai_v13_long_context.pt` and `checkpoints/storyai_v14_instruction.pt`. V8 is preserved. The instruction trainer uses `dataset/v12/pilot_expanded/train.jsonl` (16 authored examples) and validation (4 authored examples); **this is insufficient for reliable open-domain novel generation**. Add many human-reviewed, diverse examples before serious instruction training. The training script intentionally excludes unverified auto-extracted pairs.

**Generate:**

```bash
python -m story_engine.v14.novel --spec story_spec.example.json --outdir output/my_novel
```

Outputs `novel.md`, `chapter_XX.md`, `outline.json`, and `story_state.json`. The engine checks missing character names and records warnings. It does **not** falsely claim that the model followed every plot requirement or maintained all continuity. The outline is structured scaffolding; the model generates the actual prose.

## Limits / next real work

- 7.47M parameters remain the default. Larger 50M–125M *from scratch* requires a substantially larger corpus and longer pretraining, and is not included as a trained model.
- More context does not guarantee better text; evaluate raw model outputs.
- The engine carries recent scene snippets, not a semantic long-term memory extractor. For long novels, manual/editable story-state summaries and richer event tracking are needed.
- Chapter generation is sequential and can be slow because decoding recomputes attention (no KV cache).
- If CUDA runs out of memory, reduce block size / gradient accumulation / per-step batch size. Always test before expensive runs.
