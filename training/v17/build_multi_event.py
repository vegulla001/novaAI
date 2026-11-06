"""Multi-event character-grounded continuation dataset."""

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

TRAIN_NAMES = [
    "Arin", "Elara", "Kira", "Tomas", "Nila",
    "Owen", "Iris", "Caden", "Mira", "Jonas",
]
VAL_NAMES = [
    "Soren", "Veyra", "Bren", "Lyra",
    "Darian", "Neris", "Torin", "Amara",
]

PLACES = [
    "courtyard", "tower", "cellar", "garden",
    "library", "bridge", "harbor", "stable",
]

SKILLS = ["ownership", "transfer", "location", "access"]


def create_pair(rng, names, pair_id, skill, validation=False):
    a, b, c, d = rng.sample(names, 4)
    place1, place2 = rng.sample(PLACES, 2)
    rows = []

    for variant in (0, 1):
        if skill == "ownership":
            owner = b if variant == 0 else c

            facts = (
                f"{a} carried a map into the fortress. "
                f"{b} brought a lantern, and {c} brought a rope. "
                f"Before entering, {owner} picked up the iron key. "
                f"{d} asked who could unlock the gate. "
                f"Nobody passed the key to anyone else."
            )
            ending = (
                f"{owner} felt for the iron key and approached the gate. "
                f"{b} raised the lantern to illuminate the lock. "
                f"The others watched as {owner} turned the key."
            )
            expected = owner

        elif skill == "transfer":
            receiver = c if variant == 0 else d

            facts = (
                f"{a} found a brass key in the old tower. "
                f"{a} gave it to {b} before the group descended. "
                f"At the bottom of the stairs, {b} handed it to {receiver}. "
                f"{c} and {d} waited near a sealed door. "
                f"There were no further exchanges."
            )
            ending = (
                f"{receiver} examined the brass key beside the sealed door. "
                f"After checking the lock, {receiver} opened it. "
                f"{a} followed the group into the passage."
            )
            expected = receiver

        elif skill == "location":
            final_place = place1 if variant == 0 else place2

            facts = (
                f"{a} and {b} began in the castle hall. "
                f"{a} went to the courtyard to investigate a noise. "
                f"Later, {a} left the courtyard and traveled to the "
                f"{final_place}. "
                f"{b} stayed in the hall with {c}. "
                f"{d} called out when a distant bell sounded."
            )
            ending = (
                f"At the {final_place}, {a} stopped to listen to the bell. "
                f"Back in the hall, {b} and {c} wondered whether "
                f"{d} had discovered something dangerous."
            )
            expected = final_place

        else:
            # Access depends on whether the key reaches the gate.
            recipient = a if variant == 0 else c

            facts = (
                f"The northern gate was locked. "
                f"{a} stood beside it, while {b} held the only key. "
                f"{b} handed the key to {recipient}. "
                f"{c} watched from the courtyard. "
                f"No one exchanged the key again. "
                f"{a} needed to get through the gate."
            )

            if variant == 0:
                ending = (
                    f"{a} used the key to unlock the northern gate. "
                    f"With the passage open, {a} stepped through "
                    f"and called to the others."
                )
            else:
                ending = (
                    f"{a} approached the locked gate but could not open it. "
                    f"Since {c} had the key, {a} called to {c} "
                    f"and asked for help."
                )
            expected = recipient

        if validation:
            intro = (
                "Previously in the story:\n"
                "Evening shadows covered the old fortress.\n"
            )
            label = "\n\nWrite the next passage:\n"
        else:
            intro = (
                "Story so far:\n"
                "The travelers entered an ancient fortress.\n"
            )
            label = "\n\nContinuation:\n"

        rows.append({
            "id": f"{pair_id}_{variant}",
            "pair_id": pair_id,
            "skill": skill,
            "variant": variant,
            "prompt": intro + facts + label,
            "completion": ending,
            "expected": expected,
        })

    assert rows[0]["prompt"] != rows[1]["prompt"]
    assert rows[0]["completion"] != rows[1]["completion"]
    return rows


def encode(tok, row, eos_id, max_length):
    prompt_ids = tok.encode(row["prompt"]).ids
    whole = tok.encode(
        row["prompt"] + row["completion"]
    ).ids

    if whole[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer boundary mismatch: " + row["id"]
        )

    if len(whole) + 1 > max_length:
        raise ValueError(
            "Record exceeds context limit: " + row["id"]
        )

    ids = whole + [eos_id]
    labels = (
        [-100] * len(prompt_ids)
        + whole[len(prompt_ids):]
        + [eos_id]
    )

    assert len(ids) == len(labels)
    assert len(ids) >= 2
    assert any(x != -100 for x in labels[1:])

    return {"input_ids": ids, "labels": labels}


def build_split(rng, tok, names, count, prefix,
                validation, eos_id, max_length):
    encoded = []
    metadata = []

    for i in range(count):
        skill = SKILLS[i % len(SKILLS)]
        pair = create_pair(
            rng, names, f"{prefix}_{i}",
            skill, validation
        )

        for row in pair:
            encoded.append(
                encode(tok, row, eos_id, max_length)
            )
            metadata.append(row)

    return encoded, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        default="dataset/v15_prose_expanded/tokenizer.json"
    )
    ap.add_argument(
        "--out-dir",
        default="dataset/v17_multi_event"
    )
    ap.add_argument("--train-pairs", type=int, default=2400)
    ap.add_argument("--val-pairs", type=int, default=400)
    ap.add_argument("--max-length", type=int, default=1025)
    ap.add_argument("--seed", type=int, default=1702)
    args = ap.parse_args()

    if min(args.train_pairs, args.val_pairs) < 1:
        ap.error("Pair counts must be positive")

    if set(TRAIN_NAMES) & set(VAL_NAMES):
        raise ValueError("Train/validation name overlap")

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError("Missing EOS token")

    train, train_meta = build_split(
        random.Random(args.seed),
        tok, TRAIN_NAMES, args.train_pairs,
        "train", False, eos_id, args.max_length
    )

    val, val_meta = build_split(
        random.Random(args.seed + 1),
        tok, VAL_NAMES, args.val_pairs,
        "val", True, eos_id, args.max_length
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(train, out / "train_sft.pt")
    torch.save(val, out / "val_sft.pt")

    for name, rows in [
        ("train_cases.jsonl", train_meta),
        ("val_cases.jsonl", val_meta),
    ]:
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "train_examples": len(train),
        "val_examples": len(val),
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "skills": SKILLS,
        "eos_id": eos_id,
        "train_val_names_disjoint": True,
        "validation_prompt_style_held_out": True,
        "model_training_performed": False,
    }

    (out / "build_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print("MULTI-EVENT DATASET BUILT")


if __name__ == "__main__":
    main()
