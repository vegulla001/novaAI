# Story AI v16: Character guard (no training)

A **deterministic, fail-closed application fix**, not a trained character-copy capability. Provide the story's canonical characters and `safe_beats` in a scene JSON file. Without a checkpoint, it emits the approved beats with zero GPU use. With a checkpoint, it attempts one model continuation and only accepts it if it passes a conservative, incomplete surface validator. If rejected, it emits the approved beats; `generated_by_model=false` shows this plainly.

## Install
Copy `training/v16/character_guard.py` to your existing repo at that path, and `scene_castle.json` to the project root.

## Run without GPU
`PYTHONPATH=/workspace/story-ai python -m training.v16.character_guard --scene scene_castle.json --output reports/v16_character_guard.json`

## Optional one-pass model candidate
`PYTHONPATH=/workspace/story-ai python -m training.v16.character_guard --scene scene_castle.json --checkpoint checkpoints/v16_character_dynamic_01/sft_best.pt --device cuda --output reports/v16_character_guard.json`

## Important limits
- This does **not** force the neural model to copy names, and the supplied beats are authored not generated.
- The validator catches missing explicit character names, obvious unknown capitalized words, JSON contamination, and explicitly prohibited phrases. It cannot guarantee semantic consistency, grammar, or story quality. Adding a character requires updating scene JSON.
- For a trustworthy novel writer, extend this to a real state store and stronger semantic verification. This is a narrow, immediate character-identity safety barrier to stop accepting your model's bad output.
