import copy
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

BASE = Path("dataset/v17_rev3")
REV4 = Path("dataset/v17_rev4")
OUT = Path("dataset/v17_rev5")
TOK = Tokenizer.from_file(
    "dataset/v15_prose_expanded/tokenizer.json"
)

EOS = 2
OUT.mkdir(parents=True, exist_ok=True)

def names_of(case):
    chars = case["characters"]
    if isinstance(chars, dict):
        chars = chars.values()
    return list(chars)

def load_pairs(split):
    pairs = defaultdict(list)
    with (BASE / f"{split}_cases.jsonl").open(
        encoding="utf-8"
    ) as f:
        for line in f:
            c = json.loads(line)
            pairs[c["pair_id"]].append(c)

    for pair in pairs.values():
        assert len(pair) == 2
        assert {x["variant"] for x in pair} == {0, 1}
        assert pair[0]["characters"] == pair[1]["characters"]

    return list(pairs.values())

def replace_names(text, mapping):
    if not mapping:
        return text
    pattern = re.compile(
        r"\b(?:" +
        "|".join(
            re.escape(x)
            for x in sorted(mapping, key=len, reverse=True)
        ) +
        r")\b"
    )
    return pattern.sub(
        lambda match: mapping[match.group(0)], text
    )

def choose_mapping(pair, pool, rng):
    old = list(dict.fromkeys(
        name
        for case in pair
        for name in names_of(case)
    ))

    candidates = [n for n in pool if n not in old]
    rng.shuffle(candidates)

    selected = []
    prefixes = set()

    for name in candidates:
        prefix = name[:3].lower()
        if prefix not in prefixes:
            selected.append(name)
            prefixes.add(prefix)
        if len(selected) == len(old):
            break

    if len(selected) != len(old):
        raise ValueError(
            "Insufficient distinct name prefixes"
        )

    return dict(zip(old, selected))

def rename_case(case, mapping, version):
    result = copy.deepcopy(case)

    for field in (
        "context", "question", "answer",
        "fact", "plan", "expected"
    ):
        if isinstance(result.get(field), str):
            result[field] = replace_names(
                result[field], mapping
            )

    if isinstance(result["characters"], dict):
        result["characters"] = {
            k: mapping.get(v, v)
            for k, v in result["characters"].items()
        }
    else:
        result["characters"] = [
            mapping.get(v, v)
            for v in result["characters"]
        ]

    result["pair_id"] = (
        str(case["pair_id"]) + f"_rename{version}"
    )

    if "id" in result:
        result["id"] = (
            str(case["id"]) + f"_rename{version}"
        )

    for old in mapping:
        assert not re.search(
            rf"\b{re.escape(old)}\b",
            result["context"]
        )

    return result

def make_record(prompt, completion):
    p = TOK.encode(
        prompt, add_special_tokens=False
    ).ids
    a = TOK.encode(
        completion, add_special_tokens=False
    ).ids

    ids = p + a + [EOS]
    labels = [-100] * len(p) + a + [EOS]

    assert len(ids) <= 1024

    return {
        "input_ids": torch.tensor(
            ids, dtype=torch.long
        ),
        "labels": torch.tensor(
            labels, dtype=torch.long
        ),
    }

def to_records(case):
    context = case["context"]

    character_prompt = (
        context
        + "\n\nTask: Answer using the final state of the story."
        + "\nQuestion: " + case["question"]
        + "\nAnswer:\n"
    )

    reasoning_prompt = (
        context
        + "\n\nTask: Reason about the current situation"
        + " and choose a valid next action."
        + "\nPlan:\n"
    )

    return (
        make_record(character_prompt, case["answer"]),
        make_record(
            reasoning_prompt,
            "Fact: " + case["fact"]
            + "\nNext action: " + case["plan"]
        ),
    )

train_pairs = load_pairs("train")
val_pairs = load_pairs("val")

train_names = sorted({
    n for pair in train_pairs
    for case in pair for n in names_of(case)
})
val_names = sorted({
    n for pair in val_pairs
    for case in pair for n in names_of(case)
})

assert set(train_names).isdisjoint(val_names)

for split, pairs, pool, copies, seed in (
    ("train", train_pairs, train_names, 2, 1717),
    ("val", val_pairs, val_names, 1, 2727),
):
    rng = random.Random(seed)

    character = torch.load(
        REV4 / "character" / f"{split}_sft.pt",
        map_location="cpu", weights_only=True
    )
    reasoning = torch.load(
        REV4 / "reasoning" / f"{split}_sft.pt",
        map_location="cpu", weights_only=True
    )

    renamed_cases = []

    for pair in pairs:
        for version in range(copies):
            mapping = choose_mapping(pair, pool, rng)
            transformed = [
                rename_case(c, mapping, version)
                for c in pair
            ]

            a, b = sorted(
                transformed,
                key=lambda x: x["variant"]
            )

            assert a["characters"] == b["characters"]
            assert a["answer"] != b["answer"]
            assert a["context"] != b["context"]

            for case in transformed:
                char, reason = to_records(case)
                character.append(char)
                reasoning.append(reason)
                renamed_cases.append(case)

            if len(renamed_cases) == 2:
                print(f"\n{split.upper()} RENAMING EXAMPLE")
                print("MAPPING:", mapping)
                print("STORY:", a["context"])
                print("ANSWER:", a["answer"])
                print("PLAN:", a["plan"])

    for task, records in (
        ("character", character),
        ("reasoning", reasoning),
    ):
        folder = OUT / task
        folder.mkdir(parents=True, exist_ok=True)

        if split == "train":
            rng.shuffle(records)

        torch.save(
            records,
            folder / f"{split}_sft.pt"
        )
        print(
            split, task, "examples:", len(records)
        )

    with (
        OUT / f"{split}_renamed_cases.jsonl"
    ).open("w", encoding="utf-8") as f:
        for case in renamed_cases:
            f.write(
                json.dumps(case, ensure_ascii=False)
                + "\n"
            )

print("\nREV5 BUILD: PASS")
