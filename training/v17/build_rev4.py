import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

SOURCE = Path("dataset/v17_rev3")
OUTPUT = Path("dataset/v17_rev4")
TOKENIZER = Tokenizer.from_file(
    "dataset/v15_prose_expanded/tokenizer.json"
)
EOS = 2

def read_cases(split):
    with (SOURCE / f"{split}_cases.jsonl").open(
        encoding="utf-8"
    ) as f:
        return [json.loads(line) for line in f]

def repair_chronology(context):
    parts = context.split(". ")

    shows = [
        i for i, sentence in enumerate(parts)
        if " showed " in sentence
        and "silver compass" in sentence
    ]
    returns = [
        i for i, sentence in enumerate(parts)
        if " returned " in sentence
        and "silver compass" in sentence
    ]

    if len(shows) == 1 and len(returns) == 1:
        if returns[0] < shows[0]:
            sentence = parts.pop(shows[0])
            parts.insert(returns[0], sentence)

    return ". ".join(parts)

def make_example(prompt, answer):
    p = TOKENIZER.encode(
        prompt, add_special_tokens=False
    ).ids
    a = TOKENIZER.encode(
        answer, add_special_tokens=False
    ).ids

    ids = p + a + [EOS]
    labels = [-100] * len(p) + a + [EOS]

    if not (1 < len(ids) <= 1024):
        raise ValueError(
            f"Invalid sequence length: {len(ids)}"
        )

    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def original_examples(cases):
    character = []
    reasoning = []
    chronology_repairs = 0

    for case in cases:
        context = case["context"]
        corrected = repair_chronology(context)

        if corrected != context:
            chronology_repairs += 1

        character_prompt = (
            corrected
            + "\n\nTask: Answer using the final state of the story."
            + "\nQuestion: " + case["question"]
            + "\nAnswer:\n"
        )

        reasoning_prompt = (
            corrected
            + "\n\nTask: Reason about the current situation"
            + " and choose a valid next action."
            + "\nPlan:\n"
        )

        character.append(
            make_example(character_prompt, case["answer"])
        )

        reasoning.append(
            make_example(
                reasoning_prompt,
                "Fact: " + case["fact"]
                + "\nNext action: " + case["plan"]
            )
        )

    return character, reasoning, chronology_repairs

def names_from(cases):
    names = set()

    for case in cases:
        characters = case["characters"]

        if isinstance(characters, dict):
            characters = characters.values()

        names.update(
            n for n in characters if isinstance(n, str)
        )

    return sorted(names)

def scaffold(rng, names, index):
    a, b, c, d = rng.sample(names, 4)

    objects = [
        "sealed map", "silver key", "bronze seal",
        "old journal", "wooden box", "iron key"
    ]
    obj = rng.choice(objects)
    style = index % 6

    if style == 0:
        story = (
            f"The registry identified {a} as the messenger. "
            f"{b} guarded the gate while {c} waited with {d}."
        )
        task = "Copy the messenger's name exactly."
        answer = a

    elif style == 1:
        story = (
            f"{a} was holding the {obj}. "
            f"{a} handed the {obj} to {b}. "
            f"Meanwhile, {c} spoke with {d}. "
            f"No further transfer occurred."
        )
        task = f"Who currently holds the {obj}?"
        answer = f"{b} holds the {obj}."

    elif style == 2:
        story = (
            f"{a} originally carried the {obj}. "
            f"{a} gave the {obj} to {b}. "
            f"{c} met {d} by the gate. "
            f"Later, {b} passed the {obj} to {d}. "
            f"No one moved the {obj} afterward."
        )
        task = f"Who currently holds the {obj}?"
        answer = f"{d} holds the {obj}."

    elif style == 3:
        story = (
            f"{a} waited in the garden. "
            f"{b} remained at the watchtower. "
            f"{c} visited {d} at the old mill. "
            f"Later, {a} walked to the watchtower "
            f"and stayed there."
        )
        task = f"Where is {a} now?"
        answer = f"{a} is at the watchtower."

    elif style == 4:
        story = (
            f"{a} discovered that the bridge had collapsed. "
            f"{c} and {d} discussed the weather elsewhere. "
            f"{a} privately informed {b} about the bridge. "
            f"No one else was told."
        )
        task = (
            f"Who was informed about the bridge by {a}?"
        )
        answer = f"{b} was informed about the bridge."

    else:
        story = (
            f"{a} brought the {obj} to the archive. "
            f"{c} carried a different parcel. "
            f"The archivist asked {b} to safeguard the "
            f"{obj}, so {a} gave it to {b}. "
            f"{d} left the archive without touching it."
        )
        task = f"Who has the {obj} at the end?"
        answer = f"{b} holds the {obj}."

    prompt = (
        "Story so far:\n"
        + story
        + "\n\nTask: Answer using the final state of the story."
        + "\nQuestion: " + task
        + "\nAnswer:\n"
    )

    return prompt, answer

train_cases = read_cases("train")
val_cases = read_cases("val")

train_names = names_from(train_cases)
val_names = names_from(val_cases)

assert set(train_names).isdisjoint(val_names), (
    "Training and validation character names overlap"
)

for split, cases, names, extra, seed in [
    ("train", train_cases, train_names, 18000, 1717),
    ("val", val_cases, val_names, 600, 2727),
]:
    character, reasoning, repaired = original_examples(cases)

    rng = random.Random(seed)
    scaffolds = []

    for i in range(extra):
        prompt, answer = scaffold(rng, names, i)

        if split == "val" and i < 3:
            print("\nVALIDATION SCAFFOLD:")
            print(prompt)
            print("EXPECTED:", answer)

        scaffolds.append(make_example(prompt, answer))

    for case in cases:
        corrected = repair_chronology(case["context"])
        parts = corrected.split(". ")

        shows = [
            i for i, s in enumerate(parts)
            if " showed " in s and "silver compass" in s
        ]
        returns = [
            i for i, s in enumerate(parts)
            if " returned " in s and "silver compass" in s
        ]

        if len(shows) == 1 and len(returns) == 1:
            assert shows[0] < returns[0]

    (OUTPUT / "character").mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "reasoning").mkdir(
        parents=True, exist_ok=True
    )

    if split == "train":
        character.extend(scaffolds)
        rng.shuffle(character)
    else:
        torch.save(
            scaffolds, OUTPUT / "copy_val_sft.pt"
        )

    torch.save(
        character,
        OUTPUT / "character" / f"{split}_sft.pt"
    )
    torch.save(
        reasoning,
        OUTPUT / "reasoning" / f"{split}_sft.pt"
    )

    print(
        split.upper(),
        "character:", len(character),
        "reasoning:", len(reasoning),
        "chronology repairs:", repaired
    )

print("\nREV4 DATASET BUILD: PASS")
print("OUTPUT:", OUTPUT)
