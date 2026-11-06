"""Build counterfactual character-grounded prose SFT data for Story AI v17.

Produces response-only GPT labels and paired validation metadata.
No model training is performed.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer


TRAIN_NAMES = [
    "Arin", "Elara", "Kira", "Tomas", "Nila", "Owen",
    "Iris", "Caden", "Mira", "Jonas", "Dev", "Lena",
]

VAL_NAMES = [
    "Soren", "Veyra", "Bren", "Lyra",
    "Darian", "Neris", "Torin", "Amara",
]

PLACES = [
    ("courtyard", "tower"),
    ("library", "garden"),
    ("cellar", "bridge"),
    ("harbor", "market"),
    ("temple", "stable"),
]

OPENINGS = [
    "Rain tapped against the old stone walls.",
    "A cold wind moved through the deserted streets.",
    "The evening light faded behind the mountains.",
    "Thunder rolled above the ancient fortress.",
]

def make_pair(rng, names, pair_id):
    a, b, c = rng.sample(names, 3)
    opening = rng.choice(OPENINGS)
    place1, place2 = rng.choice(PLACES)
    skill = rng.choice(["ownership", "transfer", "location"])

    rows = []

    for variant in (0, 1):
        if skill == "ownership":
            owner = a if variant == 0 else b

            context = (
                f"{opening}\n"
                f"{a}, {b}, and {c} approached a locked gate. "
                f"{c} carried a lantern. "
                f"{owner} carried the iron key. "
                f"No one exchanged belongings. "
                f"The lock could only be opened with that key."
            )

            continuation = (
                f"{owner} took the iron key from a pocket and fitted it "
                f"into the lock. The gate opened with a heavy scrape. "
                f"{c} raised the lantern while the others stepped through."
            )

        elif skill == "transfer":
            recipient = b if variant == 0 else c

            context = (
                f"{opening}\n"
                f"{a} initially carried the brass key. "
                f"{b} and {c} waited beside a locked door. "
                f"{a} handed the brass key to {recipient}. "
                f"No further exchange took place. "
                f"The door required the brass key to open."
            )

            continuation = (
                f"{recipient} examined the brass key and pushed it into "
                f"the lock. The door swung inward. "
                f"{a} followed, relieved that they could finally enter."
            )

        else:
            destination = place1 if variant == 0 else place2

            context = (
                f"{opening}\n"
                f"{a} left the castle hall and walked to the {destination}. "
                f"{b} remained inside the castle hall with {c}. "
                f"No one moved again. "
                f"A bell sounded in the distance."
            )

            continuation = (
                f"At the {destination}, {a} heard the bell and stopped "
                f"to listen. Back in the castle hall, {b} asked {c} "
                f"whether the sound was a warning."
            )

        prompt = (
            "Story so far:\n"
            + context
            + "\n\nContinuation:\n"
        )

        rows.append({
            "id": f"{pair_id}_{variant}",
            "pair_id": pair_id,
            "skill": skill,
            "variant": variant,
            "prompt": prompt,
            "completion": continuation,
            "expected_actor": (
                owner if skill == "ownership"
                else recipient if skill == "transfer"
                else a
            ),
            "expected_location": (
                destination if skill == "location" else None
            ),
        })

    assert rows[0]["completion"] != rows[1]["completion"]
    return rows


def encode_row(tok, row, eos_id, max_length):
    prompt_ids = tok.encode(row["prompt"]).ids

    # Encode prompt + completion together so boundary tokenization
    # matches actual autoregressive inference.
    text_ids = tok.encode(
        row["prompt"] + row["completion"]
    ).ids

    # Verify that prompt tokenization is a prefix. Some tokenizer
    # configurations can merge tokens across the boundary.
    if text_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Tokenizer boundary mismatch for {row['id']}; "
            "do not silently train misaligned labels."
        )

    if eos_id is not None:
        text_ids = text_ids + [eos_id]

    if len(text_ids) > max_length:
        raise ValueError(
            f"Example {row['id']} requires {len(text_ids)} "
            f"tokens; maximum is {max_length}."
        )

    if len(text_ids) <= len(prompt_ids):
        raise ValueError("Completion has no tokens")

    labels = (
        [-100] * len(prompt_ids)
        + text_ids[len(prompt_ids):]
    )

    assert len(text_ids) == len(labels)

    return {
        "input_ids": text_ids,
        "labels": labels,
    }


def build_split(rng, tok, names, count, prefix, eos_id, max_length):
    encoded = []
    metadata = []
    skills = {"ownership": 0, "transfer": 0, "location": 0}

    for i in range(count):
        pair = make_pair(rng, names, f"{prefix}_{i}")
        skills[pair[0]["skill"]] += 1

        for row in pair:
            encoded.append(
                encode_row(tok, row, eos_id, max_length)
            )
            metadata.append(row)

    return encoded, metadata, skills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        default="dataset/v15_prose_expanded/tokenizer.json"
    )
    ap.add_argument(
        "--out-dir",
        default="dataset/v17_character_grounded"
    )
    ap.add_argument("--train-pairs", type=int, default=1500)
    ap.add_argument("--val-pairs", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=1025)
    ap.add_argument("--seed", type=int, default=1701)
    args = ap.parse_args()

    if args.train_pairs < 1 or args.val_pairs < 1:
        ap.error("Pair counts must be positive")

    tok = Tokenizer.from_file(args.tokenizer)
    eos_id = tok.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError("Tokenizer has no <eos> token")

    if set(TRAIN_NAMES) & set(VAL_NAMES):
        raise ValueError("Train/validation character leakage")

    train, train_meta, train_skills = build_split(
        random.Random(args.seed),
        tok,
        TRAIN_NAMES,
        args.train_pairs,
        "train",
        eos_id,
        args.max_length,
    )

    val, val_meta, val_skills = build_split(
        random.Random(args.seed + 1),
        tok,
        VAL_NAMES,
        args.val_pairs,
        "val",
        eos_id,
        args.max_length,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.save(train, out / "train_sft.pt")
    torch.save(val, out / "val_sft.pt")

    for filename, rows in (
        ("train_cases.jsonl", train_meta),
        ("val_cases.jsonl", val_meta),
    ):
        with (out / filename).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "train_examples": len(train),
        "val_examples": len(val),
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "train_skills": train_skills,
        "val_skills": val_skills,
        "eos_id": eos_id,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_val_names_disjoint": True,
        "model_training_performed": False,
    }

    (out / "build_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print("DATASET BUILD COMPLETE:", out)


if __name__ == "__main__":
    main()
