"""Story AI v17 revision-2 generation benchmark.

Tests:
  1. Character and state answers
  2. Causal planning
  3. Counterfactual pairs
  4. Genuine novel continuation length and premature EOS

The automatic scoring is diagnostic, not a complete literary evaluation.
"""

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from training.v15.model import GPT, GPTConfig


def normalize(text):
    return " ".join(
        re.findall(r"[a-z0-9]+", text.casefold())
    )


def contains_phrase(text, phrase):
    a = normalize(text)
    b = normalize(phrase)
    return bool(b) and bool(
        re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", a)
    )


def load_model(path, tokenizer_path, device):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    digest = hashlib.sha256(
        Path(tokenizer_path).read_bytes()
    ).hexdigest()

    if ckpt.get("tokenizer_sha256") != digest:
        raise ValueError("Tokenizer/checkpoint mismatch")

    cfg = GPTConfig(**ckpt["model_config"])

    if (
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
        cfg.vocab_size,
    ) != (13, 8, 512, 8192):
        raise ValueError("Expected v17 48.51M architecture")

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    return model, ckpt


def generate(
    model,
    tokenizer,
    prompt_ids,
    device,
    max_tokens,
    temperature,
    top_k,
):
    if len(prompt_ids) >= model.config.block_size:
        raise ValueError("Prompt exceeds GPT context")

    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.detach().cpu().tolist()
    else:
        prompt_ids = list(prompt_ids)

    ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        output = model.generate(
            ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=2,
        )

    new_ids = output[0, len(prompt_ids):].tolist()

    ended = 2 in new_ids
    if ended:
        new_ids = new_ids[:new_ids.index(2)]

    result = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    )

    return result, ended, len(new_ids)


def select_pairs(path, per_skill, seed):
    groups = defaultdict(list)

    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                groups[row["pair_id"]].append(row)

    buckets = defaultdict(list)

    for pair in groups.values():
        if len(pair) != 2:
            raise ValueError("Incomplete counterfactual pair")

        pair = sorted(pair, key=lambda r: r["variant"])

        if pair[0]["expected"] == pair[1]["expected"]:
            raise ValueError("Pair has identical expected outcomes")

        buckets[pair[0]["skill"]].append(pair)

    rng = random.Random(seed)
    selected = []

    for skill in sorted(buckets):
        bucket = sorted(
            buckets[skill],
            key=lambda group: group[0]["pair_id"],
        )
        rng.shuffle(bucket)
        selected.extend(bucket[:per_skill])

    return selected


def evaluate_cases(
    model,
    tokenizer,
    device,
    args,
    records,
):
    pairs = select_pairs(
        args.cases,
        args.pairs_per_skill,
        args.seed,
    )

    pair_results = []

    for pair in pairs:
        character_scores = []
        plan_scores = []
        answer_texts = []

        for case in pair:
            # Factual task: greedy decoding avoids stochastic
            # differences when comparing counterfactual variants.
            char_text, char_eos, char_tokens = generate(
                model,
                tokenizer,
                tokenizer.encode(
                    case["character_prompt"]
                ).ids,
                device,
                args.answer_tokens,
                1.0,
                1,
            )

            # Match the full expected answer, not merely a name
            # that might appear somewhere in unrelated prose.
            char_correct = contains_phrase(
                char_text,
                case["answer"],
            )

            plan_text, plan_eos, plan_tokens = generate(
                model,
                tokenizer,
                tokenizer.encode(
                    case["reasoning_prompt"]
                ).ids,
                device,
                args.plan_tokens,
                1.0,
                1,
            )

            # This exact-reference match is deliberately strict;
            # a semantically valid paraphrase may score False.
            plan_correct = contains_phrase(
                plan_text,
                case["plan"],
            )

            character_scores.append(char_correct)
            plan_scores.append(plan_correct)
            answer_texts.append(normalize(char_text))

            records.append({
                "test": "grounding",
                "case_id": case["pair_id"],
                "variant": case["variant"],
                "skill": case["skill"],
                "expected_answer": case["answer"],
                "generated_answer": char_text,
                "answer_correct": char_correct,
                "answer_eos": char_eos,
                "answer_tokens": char_tokens,
                "expected_plan": case["plan"],
                "generated_plan": plan_text,
                "plan_correct": plan_correct,
                "plan_eos": plan_eos,
                "plan_tokens": plan_tokens,
            })

            print(
                "CASE",
                case["pair_id"],
                case["variant"],
                case["skill"],
                "answer_correct=", char_correct,
                "plan_correct=", plan_correct,
                flush=True,
            )

        pair_results.append({
            "skill": pair[0]["skill"],
            "both_answers_correct": all(character_scores),
            "both_plans_correct": all(plan_scores),
            "answers_changed": answer_texts[0] != answer_texts[1],
        })

    grounding = [
        r for r in records
        if r["test"] == "grounding"
    ]

    return {
        "examples": len(grounding),
        "pairs": len(pair_results),
        "character_accuracy": sum(
            r["answer_correct"] for r in grounding
        ) / len(grounding),
        "exact_plan_accuracy": sum(
            r["plan_correct"] for r in grounding
        ) / len(grounding),
        "both_answers_correct_pairs": sum(
            p["both_answers_correct"] for p in pair_results
        ) / len(pair_results),
        "both_plans_correct_pairs": sum(
            p["both_plans_correct"] for p in pair_results
        ) / len(pair_results),
        "answers_changed_pairs": sum(
            p["answers_changed"] for p in pair_results
        ) / len(pair_results),
    }


def evaluate_prose(
    model,
    tokenizer,
    device,
    args,
    records,
):
    rows = torch.load(
        args.prose_val,
        map_location="cpu",
        weights_only=True,
    )

    rng = random.Random(args.seed + 1)
    indices = rng.sample(
        range(len(rows)),
        min(args.prose_samples, len(rows)),
    )

    metrics = []

    for index in indices:
        row = rows[index]

        ids = row["input_ids"]
        labels = row["labels"]

        first_target = next(
            i for i, value in enumerate(labels)
            if value != -100
        )

        # Use the original tokenized prose prompt exactly.
        prompt_ids = ids[:first_target]

        text, ended, token_count = generate(
            model,
            tokenizer,
            prompt_ids,
            device,
            args.prose_tokens,
            args.temperature,
            40,
        )

        word_count = len(
            re.findall(r"\b[\w']+\b", text)
        )

        format_clean = not any(
            marker in text
            for marker in (
                "Current fact:",
                "Next action:",
                "<END_ANSWER>",
                "\nPlan:",
            )
        )

        metrics.append({
            "ended": ended,
            "tokens": token_count,
            "words": word_count,
            "format_clean": format_clean,
        })

        records.append({
            "test": "prose",
            "dataset_index": index,
            "prompt_tail": tokenizer.decode(
                prompt_ids[-120:].tolist()
                if isinstance(prompt_ids, torch.Tensor)
                else list(prompt_ids[-120:]),
                skip_special_tokens=True,
            ),
            "generated": text,
            "ended_with_eos": ended,
            "generated_tokens": token_count,
            "generated_words": word_count,
            "format_clean": format_clean,
        })

        print(
            "PROSE",
            index,
            "words=", word_count,
            "eos=", ended,
            "format_clean=", format_clean,
            flush=True,
        )

    n = len(metrics)

    return {
        "examples": n,
        "mean_words": sum(
            m["words"] for m in metrics
        ) / n,
        "mean_tokens": sum(
            m["tokens"] for m in metrics
        ) / n,
        "premature_eos_rate": sum(
            m["ended"] for m in metrics
        ) / n,
        "format_clean_rate": sum(
            m["format_clean"] for m in metrics
        ) / n,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)

    ap.add_argument(
        "--tokenizer",
        default="dataset/v15_prose_expanded/tokenizer.json",
    )
    ap.add_argument(
        "--cases",
        default="dataset/v17_rev2/val_cases.jsonl",
    )
    ap.add_argument(
        "--prose-val",
        default="dataset/v15_prose_expanded/continuation_sft/val_sft.pt",
    )
    ap.add_argument(
        "--out",
        default="checkpoints/v17_rev2/evaluation.jsonl",
    )

    ap.add_argument("--pairs-per-skill", type=int, default=3)
    ap.add_argument("--prose-samples", type=int, default=6)
    ap.add_argument("--answer-tokens", type=int, default=64)
    ap.add_argument("--plan-tokens", type=int, default=120)
    ap.add_argument("--prose-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1708)
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = ap.parse_args()

    if min(
        args.pairs_per_skill,
        args.prose_samples,
        args.answer_tokens,
        args.plan_tokens,
        args.prose_tokens,
    ) <= 0:
        ap.error("Test counts and token limits must be positive")

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer)

    model, ckpt = load_model(
        args.checkpoint,
        args.tokenizer,
        device,
    )

    records = []

    grounding = evaluate_cases(
        model,
        tokenizer,
        device,
        args,
        records,
    )

    prose = evaluate_prose(
        model,
        tokenizer,
        device,
        args,
        records,
    )

    output = Path(args.out)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": ckpt.get("step"),
        "grounding": grounding,
        "prose": prose,
    }

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nREV2 GENERATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print("OUTPUT:", output)
    print("SUMMARY:", summary_path)


if __name__ == "__main__":
    main()
