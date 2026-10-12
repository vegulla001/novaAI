# StoryAI V9 — from-scratch conversational action model (experimental)

This is a **small domain-specific proof of concept**, not a general chatbot. It trains a new byte-level causal Transformer from random initialization (no pretrained weights, APIs, or external AI). It learns to map short story-editing requests to JSON actions. The deterministic story engine validates and persists actions. It does **not** generate or rewrite chapter prose, support pronoun resolution, or understand arbitrary language yet.

Requires Python 3.10+ and `torch` (`pip install torch`). Run commands from the directory containing `conversation/` and `story_engine/`:

```bash
python -m conversation.build_dataset --count 12000
python -m conversation.train --steps 1000 --batch-size 16 --block-size 256
python -m conversation.chat
```

CPU training can take a long time; for a quick syntax/smoke test run `python -m conversation.train --steps 2 --batch-size 2 --eval-every 1`. A smoke-test checkpoint is not useful for chat.

Chat commands:

```
/new The Lost Kingdom
Add a character named Lyra who is a mage.
Make Lyra a villain.
Add a chapter called The Betrayal.
Undo the last action.
/show
/quit
```

Caveats: synthetic dataset is small and formulaic, and random row splitting leaks near-identical templates between training and validation. Reported validation loss is **not** a measure of robust language understanding. A low loss does not imply generalization. The model may output malformed JSON or incorrect actions; the engine only accepts a fixed allowlist and validates fields. Review `Action >` before trusting edits. Story data is saved to `stories/`. For serious use, add human confirmation, broader data, entity resolution, per-story context, tests on unseen templates, and narrative prose training.
