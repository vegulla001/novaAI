import json
import os
import torch

from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig

DEVICE = "cpu"
CHECKPOINT = "checkpoints/v17_rev5_pilot/sft_best.pt"
CASES = "dataset/v17_rev5/val_renamed_cases.jsonl"

torch.set_num_threads(min(8, os.cpu_count() or 1))

print("LOADING MODEL ON CPU...", flush=True)

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

model = GPT(GPTConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

tok = Tokenizer.from_file(
    "dataset/v15_prose_expanded/tokenizer.json"
)

@torch.inference_mode()
def score_name(prompt_ids, name):
    name_ids = tok.encode(
        name,
        add_special_tokens=False
    ).ids

    if not name_ids:
        return float("-inf")

    full = prompt_ids + name_ids[:-1]

    x = torch.tensor(
        [full],
        dtype=torch.long,
        device=DEVICE
    )

    logits, _ = model(x, x)

    start = len(prompt_ids) - 1

    selected = logits[
        0,
        start:start + len(name_ids)
    ]

    log_probs = torch.log_softmax(
        selected.float(),
        dim=-1
    )

    targets = torch.tensor(
        name_ids,
        dtype=torch.long,
        device=DEVICE
    )

    scores = log_probs.gather(
        -1,
        targets[:, None]
    ).squeeze(-1)

    return scores.mean().item()

def characters(case):
    chars = case["characters"]

    if isinstance(chars, dict):
        return list(chars.values())

    return list(chars)

with open(CASES, encoding="utf-8") as f:
    cases = [json.loads(line) for line in f]

for skill in ("knowledge", "ownership", "location"):
    subset = [
        c for c in cases
        if c["skill"] == skill
    ][:30]

    correct = 0
    tested = 0

    print("\n" + "=" * 65)
    print("SKILL:", skill.upper())

    for case in subset:
        # Location predictions are places, not character names.
        if skill == "location":
            continue

        expected = case["expected"]
        names = sorted(set(characters(case)))

        if expected not in names:
            print("SKIPPED:", expected)
            continue

        prompt = (
            case["context"]
            + "\n\nTask: Answer using the final state of the story."
            + "\nQuestion: "
            + case["question"]
            + "\nAnswer:\n"
        )

        prompt_ids = tok.encode(
            prompt,
            add_special_tokens=False
        ).ids

        scores = [
            (name, score_name(prompt_ids, name))
            for name in names
        ]

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        predicted = scores[0][0]
        correct += int(predicted == expected)
        tested += 1

        print("\nEXPECTED:", expected)
        print("SELECTED:", predicted)
        print("TOP THREE:", scores[:3], flush=True)

    if tested:
        print(
            f"\nRESULT {skill}: {correct}/{tested}",
            flush=True
        )

print("\nCPU CHARACTER-SELECTION TEST COMPLETE")
print("NO TRAINING PERFORMED")
