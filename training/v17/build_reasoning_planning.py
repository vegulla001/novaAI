"""Story AI v17: causal reasoning, planning, and GPT prose targets."""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import torch
from tokenizers import Tokenizer

TRAIN_NAMES = [
    "Arin", "Elara", "Kira", "Tomas", "Nila",
    "Owen", "Iris", "Caden", "Mira", "Jonas"
]

VAL_NAMES = [
    "Soren", "Veyra", "Bren", "Lyra",
    "Darian", "Neris", "Torin", "Amara"
]

SKILLS = ("transfer", "location", "access", "knowledge")


def make_pair(rng, names, pair_id, skill, validation=False):
    a, b, c, d = rng.sample(names, 4)
    gate = rng.choice(["northern gate", "stone archway", "cellar door"])
    place = rng.choice(["tower", "library", "courtyard"])
    rows = []

    for variant in (0, 1):
        if skill == "transfer":
            holder = b if variant == 0 else c

            scene = (
                f"{a} found the only iron key. "
                f"{a} gave the key to {b}. "
                f"Before reaching the locked {gate}, "
                f"{b} handed the key to {holder}. "
                f"{c} watched the exchange. "
                f"No other transfer occurred. "
                f"{a} wanted to open the {gate}."
            )

            fact = f"{holder} currently holds the key."
            constraint = f"{a} does not hold the key."
            action = f"{a} asks {holder} for the key."

            prose = (
                f"{a} stopped at the {gate} and searched for the key. "
                f"Remembering the final exchange, {a} turned to "
                f"{holder} and asked for it. "
                f"The others waited beside the silent doorway."
            )

        elif skill == "location":
            destination = place if variant == 0 else "stable"

            scene = (
                f"{a} and {b} entered the castle hall. "
                f"{a} walked to the {place}. "
                f"Later, {a} traveled to the {destination}. "
                f"{b} remained in the hall beside {c}. "
                f"No one moved after that. "
                f"{d} heard a bell and wanted to find {a}."
            )

            fact = f"{a} is currently at the {destination}."
            constraint = f"{a} is no longer in the castle hall."
            action = f"{d} searches for {a} at the {destination}."

            prose = (
                f"{d} left the hall and headed toward the "
                f"{destination}. "
                f"There, {d} spotted {a} beside the old stonework. "
                f"The distant bell sounded again."
            )

        elif skill == "access":
            holder = a if variant == 0 else c

            scene = (
                f"The {gate} was locked. "
                f"{a} needed to pass through it. "
                f"{b} held the only working key at first. "
                f"{b} handed it to {holder}. "
                f"{c} waited nearby. "
                f"The key was not transferred again."
            )

            fact = f"{holder} holds the working key."
            constraint = (
                f"{a} can open the gate directly."
                if holder == a else
                f"{a} must obtain the key or request help."
            )

            if holder == a:
                action = f"{a} unlocks the gate."
                prose = (
                    f"{a} slipped the key into the lock and turned it. "
                    f"The {gate} opened, allowing the group to enter."
                )
            else:
                action = f"{a} asks {c} to unlock the gate."
                prose = (
                    f"{a} reached the locked {gate} but had no key. "
                    f"Turning to {c}, {a} asked for help. "
                    f"{c} stepped toward the lock."
                )

        else:
            informed = b if variant == 0 else c
            uninformed = c if variant == 0 else b

            scene = (
                f"{a} discovered that the old bridge had collapsed. "
                f"{a} privately told {informed} about the collapse. "
                f"{uninformed} was away and did not hear the warning. "
                f"No one sent another message. "
                f"{d} planned to lead the group across the bridge."
            )

            fact = f"{informed} knows the bridge has collapsed."
            constraint = (
                f"{uninformed} has not been informed of the collapse."
            )
            action = f"{informed} warns {d} before the crossing."

            prose = (
                f"As {d} prepared to lead the travelers onward, "
                f"{informed} hurried forward. "
                f"The bridge had collapsed, and {informed} "
                f"warned {d} that they needed another route."
            )

        if validation:
            intro = (
                "Earlier events in this story:\n"
                "A storm was approaching the castle.\n"
            )
        else:
            intro = (
                "Story so far:\n"
                "The travelers were exploring an old castle.\n"
            )

        context = intro + scene

        plan = (
            f"Current fact: {fact}\n"
            f"Constraint: {constraint}\n"
            f"Next action: {action}"
        )

        rows.append({
            "id": f"{pair_id}_{variant}",
            "pair_id": pair_id,
            "skill": skill,
            "variant": variant,
            "context": context,
            "plan": plan,
            "prose": prose,
            "expected_fact": fact,
            "expected_action": action,
        })

    assert rows[0]["plan"] != rows[1]["plan"]
    assert rows[0]["prose"] != rows[1]["prose"]
    return rows


def encode_example(tok, prompt, completion, eos, max_length):
    # Encode separately to preserve the exact prompt token prefix
    # used at inference. Disable BOS insertion in the completion.
    p = tok.encode(prompt).ids
    y = tok.encode(completion, add_special_tokens=False).ids

    ids = p + y + [eos]
    labels = [-100] * len(p) + y + [eos]

    if len(ids) > max_length:
        raise ValueError(
            f"Record has {len(ids)} tokens; limit {max_length}"
        )

    assert len(ids) == len(labels)
    assert len(p) > 0 and len(y) > 0
    assert labels[-1] == eos
    assert all(0 <= t < tok.get_vocab_size() for t in ids)

    return {"input_ids": ids, "labels": labels}


def build(rng, tok, names, pair_count, split, eos, max_length):
    records = []
    cases = []
    counts = Counter()

    for i in range(pair_count):
        skill = SKILLS[i % len(SKILLS)]

        for row in make_pair(
            rng, names, f"{split}_{i}", skill,
            validation=(split == "val")
        ):
            # Task 1: Explicit, checkable plan followed by prose.
            prompt = (
                row["context"] +
                "\n\nTask: Plan the next event and write the scene.\n"
            )
            completion = (
                "Plan:\n" + row["plan"] +
                "\n\nProse:\n" + row["prose"]
            )

            records.append(
                encode_example(
                    tok, prompt, completion, eos, max_length
                )
            )
            counts["plan_and_prose"] += 1

            # Task 2: Novel-only output. No reasoning markers
            # appear in this task's completion.
            novel_prompt = (
                row["context"] +
                "\n\nTask: Continue the novel in prose.\n"
            )

            records.append(
                encode_example(
                    tok, novel_prompt, row["prose"],
                    eos, max_length
                )
            )
            counts["prose_only"] += 1

            cases.append(row)

    return records, cases, dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        default="dataset/v15_prose_expanded/tokenizer.json"
    )
    ap.add_argument(
        "--out-dir",
        default="dataset/v17_reasoning_planning"
    )
    ap.add_argument("--train-pairs", type=int, default=2000)
    ap.add_argument("--val-pairs", type=int, default=400)
    ap.add_argument("--max-length", type=int, default=1025)
    ap.add_argument("--seed", type=int, default=1703)
    args = ap.parse_args()

    if min(args.train_pairs, args.val_pairs) < 1:
        ap.error("Pair counts must be positive")

    if set(TRAIN_NAMES) & set(VAL_NAMES):
        raise ValueError("Character-name leakage")

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    if eos is None:
        raise ValueError("Missing EOS token")

    train, train_cases, train_counts = build(
        random.Random(args.seed), tok, TRAIN_NAMES,
        args.train_pairs, "train", eos, args.max_length
    )

    val, val_cases, val_counts = build(
        random.Random(args.seed + 1), tok, VAL_NAMES,
        args.val_pairs, "val", eos, args.max_length
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(train, out / "train_sft.pt")
    torch.save(val, out / "val_sft.pt")

    for name, cases in (
        ("train_cases.jsonl", train_cases),
        ("val_cases.jsonl", val_cases),
    ):
        with (out / name).open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    report = {
        "train_examples": len(train),
        "val_examples": len(val),
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "train_tasks": train_counts,
        "val_tasks": val_counts,
        "skills": list(SKILLS),
        "eos_id": eos,
        "training_performed": False
    }

    (out / "build_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print("REASONING AND PLANNING DATASET BUILT")


if __name__ == "__main__":
    main()
