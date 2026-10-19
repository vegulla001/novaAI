# StoryAI rented GPU setup (RunPod / compatible Linux NVIDIA instance)

## Scope
This bundle contains your source, structured training tokens, and V13 context continuation scripts. **It does not contain your V8 checkpoint**, which was excluded from the uploaded source. Upload `storyai_v8_best.pt` to `checkpoints/` separately. No external pretrained model is downloaded.

The context experiment extends **512 to 2048 tokens** using RoPE, retaining the V8 ~7.47M parameters. It does **not** train a 50M-125M model or fix instruction-following. The instruction dataset remains insufficient; do not spend a long rental session expecting novel-quality output from this stage.

## Recommended rental
Choose an NVIDIA GPU with at least 24 GB VRAM (RTX 4090 class), a PyTorch CUDA image, and a **persistent volume** (e.g., 50 GB or more). Pricing/availability depend on provider. Ensure persistent storage is mounted and save checkpoints there before stopping the pod.

## Linux commands
```bash
cd /workspace
# Upload and extract StoryAI_GPU_Ready.zip here, or into a persistent subfolder.
unzip StoryAI_GPU_Ready.zip -d story-ai
cd story-ai
# Upload your original V8 checkpoint to checkpoints/storyai_v8_best.pt
python -m pip install -r requirements-gpu.txt
# The provider image must already contain CUDA-enabled PyTorch; verify:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m training.v13.gpu_preflight --require-gpu --data-dir dataset/processed_structured --block-size 2048
BLOCK_SIZE=2048 BATCH_SIZE=1 GRAD_ACCUM=16 STEPS=1000 bash run_gpu.sh
```

For a quick smoke test **before spending hours**:
```bash
BLOCK_SIZE=1024 BATCH_SIZE=1 GRAD_ACCUM=2 STEPS=2 bash run_gpu.sh
```
**Warning:** the smoke test writes the same V13 output path; move/rename that output before full training if you want to keep it. Training a 2048-token model is a separate run; it does not resume the smoke test.

## Files and storage
- `checkpoints/storyai_v8_best.pt`: **you must upload** your existing trusted baseline.
- `dataset/processed_structured/`: included token arrays and tokenizer.
- `checkpoints/storyai_v13_long_context.pt`: generated output; keep on persistent storage.
- `logs/v13_gpu.log`: training log.
- `output/v13_long_context/report.json`: evaluation report.

Check tokenizer compatibility via preflight. If it fails, stop: the V8 checkpoint may use a different tokenizer/dataset. Do not bypass the hash check.

## GPU OOM
Reduce `BATCH_SIZE` to 1; for a 2048-token run, keep it at 1. If needed use `BLOCK_SIZE=1024` instead. A 4090 24GB should be more than enough for this small model under typical settings, but actual memory use depends on the environment.

## Next stages (not yet implemented)
1. Audit/deduplicate and expand genuine fiction training data.
2. Train a larger model from scratch only after confirming enough tokens and compute budget.
3. Train on reviewed instruction-to-scene pairs, then build a story bible/chapter planner.
4. Evaluate **raw** model output for plot adherence and prose quality; never count fallback as model success.
