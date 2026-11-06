"""Evaluate actual Story AI v17 GPT generations.

Uses held-out reasoning/planning cases. Heuristic metrics are diagnostic,
not proof of semantic reasoning or literary quality.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig


def words(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def contains_word(text, word):
    return re.search(r"\b" + re.escape(word) + r"\b", text, re.I) is not None


def assess(case, generated, mode):
    action = case["expected_action"]
    skill = case["skill"]
    parts = action.split()
    actor = parts[0]

    actor_present = contains_word(generated, actor)
    target_present = False
    event_present = False

    if skill == "transfer":
        # Expected: A asks HOLDER for the key
        target_present = contains_word(generated, parts[2])
        event_present = bool(re.search(r"\bask(?:s|ed|ing)?\b", generated, re.I))
    elif skill == "location":
        # Expected: D searches for A at DESTINATION
        target_present = contains_word(generated, parts[3])
        destination = action.split(" at the ")[-1]
        event_present = destination.lower() in generated.lower()
    elif skill == "access":
        if "unlocks" in action:
            target_present = True
            event_present = bool(re.search(r"\b(unlock|unlocked|opens|opened)\b", generated, re.I))
        else:
            target_present = contains_word(generated, parts[2])
            event_present = bool(re.search(r"\b(ask|asks|asked|help)\b", generated, re.I))
    elif skill == "knowledge":
        # Expected: INFORMED warns D before the crossing
        target_present = contains_word(generated, parts[2])
        event_present = bool(re.search(r"\b(warn|warns|warned|warning)\b", generated, re.I))

    prose = generated.split("Prose:", 1)[-1] if mode == "plan" else generated
    tokens = words(prose)
    triples = list(zip(tokens, tokens[1:], tokens[2:]))
    distinct_3gram = (
        len(set(triples)) / len(triples)
        if triples else 0.0
    )

    format_clean = (
        "Plan:" not in generated and
        "Current fact:" not in generated and
        "<END_ANSWER>" not in generated
    ) if mode == "prose" else "<END_ANSWER>" not in generated

    return {
        "actor_present": actor_present,
        "target_present": target_present,
        "event_cue_present": event_present,
        "action_cue_all": actor_present and target_present and event_present,
        "format_clean": format_clean,
        "distinct_3gram_ratio": round(distinct_3gram, 4),
        "output_words": len(tokens),
    }


def load_checkpoint(path, tokenizer_path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actual_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    if ckpt.get("tokenizer_sha256") != actual_sha:
        raise ValueError("Checkpoint/tokenizer SHA256 mismatch")

    cfg = GPTConfig(**ckpt["model_config"])
    if (cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.vocab_size) != (13, 8, 512, 8192):
        raise ValueError("Not the intended v17 48.51M architecture")

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--tokenizer",
        default="dataset/v15_prose_expanded/tokenizer.json"
    )
    p.add_argument(
        "--cases",
        default="dataset/v17_reasoning_planning/val_cases.jsonl"
    )
    p.add_argument(
        "--out",
        default="checkpoints/v17_generation_eval.jsonl"
    )
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=1705)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()

    if a.pairs <= 0 or a.max_new_tokens <= 0:
        p.error("--pairs and --max-new-tokens must be positive")

    device = (
        "cuda" if a.device == "auto" and torch.cuda.is_available()
        else "cpu" if a.device == "auto" else a.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(a.seed)

    tokenizer_path = Path(a.tokenizer)
    tok = Tokenizer.from_file(str(tokenizer_path))
    model, ckpt = load_checkpoint(a.checkpoint, tokenizer_path, device)

    cases = [
        json.loads(line)
        for line in Path(a.cases).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    groups = {}
    for case in cases:
        groups.setdefault(case["pair_id"], []).append(case)

    rng = random.Random(a.seed)
    valid = [group for group in groups.values() if len(group) == 2]
    valid.sort(key=lambda g: g[0]["pair_id"])
    rng.shuffle(valid)

    selected = valid[:a.pairs]
    if not selected:
        raise ValueError("No complete counterfactual pairs available")

    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for group in selected:
        for case in sorted(group, key=lambda c: c["variant"]):
            for mode in ("prose", "plan"):
                instruction = (
                    "\n\nTask: Continue the novel in prose.\n"
                    if mode == "prose" else
                    "\n\nTask: Plan the next event and write the scene.\n"
                )

                prompt = case["context"] + instruction
                prefix = tok.encode(prompt).ids
                if len(prefix) >= model.config.block_size:
                    raise ValueError("Prompt exceeds model context window")

                idx = torch.tensor([prefix], dtype=torch.long, device=device)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        idx,
                        max_new_tokens=a.max_new_tokens,
                        temperature=a.temperature,
                        top_k=40,
                        eos_id=2
                    )[0, len(prefix):].tolist()

                if 2 in generated_ids:
                    generated_ids = generated_ids[:generated_ids.index(2)]

                text = tok.decode(generated_ids, skip_special_tokens=True)
                metrics = assess(case, text, mode)

                result = {
                    "case_id": case["id"],
                    "pair_id": case["pair_id"],
                    "skill": case["skill"],
                    "variant": case["variant"],
                    "mode": mode,
                    "expected_action": case["expected_action"],
                    "generated": text,
                    "metrics": metrics,
                }
                results.append(result)

                print(
                    case["id"], mode,
                    "action_cue=", metrics["action_cue_all"],
                    "format_clean=", metrics["format_clean"],
                    "words=", metrics["output_words"],
                    flush=True
                )

    with output.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {}
    for mode in ("prose", "plan"):
        subset = [r for r in results if r["mode"] == mode]
        summary[mode] = {
            "examples": len(subset),
            "action_cue_rate": round(
                sum(r["metrics"]["action_cue_all"] for r in subset)
                / len(subset), 4
            ),
            "format_clean_rate": round(
                sum(r["metrics"]["format_clean"] for r in subset)
                / len(subset), 4
            ),
            "average_distinct_3gram": round(
                sum(r["metrics"]["distinct_3gram_ratio"] for r in subset)
                / len(subset), 4
            ),
            "average_output_words": round(
                sum(r["metrics"]["output_words"] for r in subset)
                / len(subset), 2
            ),
        }

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({
            "checkpoint": str(a.checkpoint),
            "checkpoint_step": ckpt.get("step"),
            "seed": a.seed,
            "temperature": a.temperature,
            "max_new_tokens": a.max_new_tokens,
            "results": summary,
        }, indent=2),
        encoding="utf-8"
    )

    print("SUMMARY:", json.dumps(summary, indent=2))
    print("GENERATIONS:", output)
    print("SUMMARY FILE:", summary_path)


if __name__ == "__main__":
    main()
