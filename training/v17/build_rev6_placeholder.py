import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

SOURCE = Path("dataset/v17_rev3")
OUT = Path("dataset/v17_rev6_placeholder")
TOK = Tokenizer.from_file(
    "dataset/v15_prose_expanded/tokenizer.json"
)
EOS = 2
BLOCK_SIZE = 1024
MIN_WORDS = 180

def names_of(case):
    value = case["characters"]
    if isinstance(value, dict):
        value = value.values()
    return list(dict.fromkeys(
        name for name in value if isinstance(name, str)
    ))

def role_mapping(pair, rng):
    names = names_of(pair[0])
    assert set(names) == set(names_of(pair[1]))
    rng.shuffle(names)
    return {
        name: f"<CHAR_{i + 1}>"
        for i, name in enumerate(names)
    }

def replace_names(text, mapping):
    pattern = re.compile(
        r"\b(?:" +
        "|".join(
            re.escape(name)
            for name in sorted(mapping, key=len, reverse=True)
        ) +
        r")\b"
    )
    return pattern.sub(
        lambda match: mapping[match.group()], text
    )

def long_scene(case, roles, rng):
    skill = case["skill"]
    names = names_of(case)
    expected = case["expected"]

    if skill == "ownership":
        holder = roles[expected]
        requester_name = case["plan"].split()[0]
        requester = roles[requester_name]

        match = re.search(
            r"holds the (.+?)\?",
            case["question"],
            re.IGNORECASE
        )
        assert match, case["question"]
        item = match.group(1)

        middle = roles[names[1]]

        return [
            f"""{holder} kept the {item} close as evening settled over the archive. The last of the daylight lay across the stone floor, and the rooms beyond the entrance were already being locked. Although several people had handled different objects during the afternoon, the {item} had ended up with {holder}. There had been no further exchange, and its whereabouts now mattered more than anyone had expected.""",

            f"""{requester} returned to the entrance after discovering that the archive could not be opened without the {item}. At first the earlier movements seemed difficult to untangle. The object had passed through {middle}'s hands, but that was not where the trail ended. Thinking through the sequence of events, {requester} remembered the later transfer to {holder} and decided to follow that lead rather than waste time questioning everyone again.""",

            f"""Nearby, the conversation had turned to a different object that had also changed hands. It distracted the witnesses because they remembered seeing it passed back and forth. Yet the two exchanges were unrelated. The person holding the other object could not help with the locked archive, and {requester} understood that confusing the objects would only delay the search. The important question was what had happened to the {item} most recently.""",

            f"""{requester} approached {holder} and explained why the archive door had to be opened before nightfall. {holder} listened without interrupting, then brought the {item} into view. The discovery relieved the immediate uncertainty, although it did not settle what should happen next. There were reasons to be cautious about allowing the object to leave its current keeper, especially while other people were still arguing over the earlier exchanges.""",

            f"""For the moment, {holder} retained possession of the {item}. {requester} suggested that they inspect the archive together instead of handing it over in the crowded entrance. They agreed to examine the lock before making another transfer. As the last light disappeared from the windows, the two moved toward the sealed doorway, with the earlier confusion finally behind them and a more difficult decision waiting inside."""
        ]

    if skill == "knowledge":
        informed = roles[expected]
        discoverer = roles[names[0]]

        match = re.search(
            r"should warn (\w+)",
            case["plan"]
        )
        assert match, case["plan"]
        leader = roles[match.group(1)]

        return [
            f"""{informed} had been troubled by the warning ever since speaking with {discoverer}. The eastern bridge had collapsed, but the travelers were still preparing to use it. From the road, the river looked peaceful enough to encourage a careless crossing. Nothing in the quiet landscape suggested the danger ahead, and that made the information even more urgent. Someone had to act before the group reached the broken span.""",

            f"""The earlier conversation had been private. {discoverer} had explained what had happened, but no general warning had followed. Other members of the party continued discussing provisions and the journey as though the original route remained safe. {informed} remembered every detail of the message and understood that silence would leave the others exposed. The discovery could no longer remain a secret held by only a few people.""",

            f"""{leader} was making final arrangements near the front of the traveling party. The supplies had been counted, the animals were ready, and several impatient travelers wanted to depart immediately. {informed} moved through the gathering crowd, searching for a moment to speak without causing unnecessary panic. It would be better to explain the danger clearly than to shout a warning that might be misunderstood.""",

            f"""When {informed} finally reached {leader}, the explanation came quickly. The bridge was no longer safe, and any attempt to cross it could end in disaster. {leader} first questioned the information because it conflicted with the plans made earlier. But {informed} repeated what {discoverer} had reported, explaining why the warning could not be ignored. The conversation changed the mood of the gathering almost immediately.""",

            f"""The departure was postponed while the travelers considered another route. Some were frustrated by the delay, while others were relieved that the danger had been discovered before anyone reached the river. {informed} remained nearby to answer questions, aware that the next decision would affect everyone. What had begun as a private warning had become the event that determined the direction of the entire journey."""
        ]

    if skill == "location":
        match = re.search(
            r"Where is (\w+) now",
            case["question"]
        )
        assert match, case["question"]
        traveler = roles[match.group(1)]
        seeker = roles[case["plan"].split()[0]]
        place = expected

        return [
            f"""At the {place}, {traveler} listened to the distant sounds of the settlement and considered the journey that had brought them there. Several places had been visited earlier in the day, each leaving behind a different memory, but the traveling had finally stopped. The quiet was welcome after so much movement. For the first time in hours, {traveler} could think without having to decide where to go next.""",

            f"""Elsewhere, {seeker} was trying to reconstruct the route from the accounts of people who had seen {traveler} pass. Some remembered the earlier visit to the hall, while others spoke of the courtyard. Those reports were not necessarily wrong, but they described older stages of the journey. {seeker} realized that following the locations in the wrong order could lead to another long and pointless search.""",

            f"""The final report pointed toward the {place}. {seeker} considered whether to trust it, then remembered that the last witness had seen no further departure. That detail mattered. A person's presence at one location earlier in the day did not mean they had remained there, but the latest confirmed movement offered a reasonable place to begin. With that distinction settled, {seeker} set off.""",

            f"""By the time {seeker} reached the {place}, the light had begun to fade. The building seemed quiet from outside, and there was a moment of uncertainty before anyone answered. Then {traveler} appeared, surprised to find someone searching so urgently. The meeting confirmed that the latest account of the journey had been correct. They spoke for a moment about the earlier stops and the confusion those movements had caused.""",

            f"""Neither wanted to begin another journey immediately. {traveler} stayed at the {place} while {seeker} explained the reason for the search. There would be decisions to make soon, and perhaps another difficult road to follow, but the immediate problem was resolved. For now they could speak face to face, with no need to rely on uncertain messages from people who had witnessed only part of the day's events."""
        ]

    raise ValueError(f"Unknown skill: {skill}")

def encode_record(prompt, paragraphs):
    p = TOK.encode(
        prompt, add_special_tokens=False
    ).ids

    for count in range(len(paragraphs), 2, -1):
        completion = "\n\n".join(paragraphs[:count])
        if len(completion.split()) < MIN_WORDS:
            continue

        a = TOK.encode(
            completion, add_special_tokens=False
        ).ids
        ids = p + a + [EOS]

        if len(ids) <= BLOCK_SIZE:
            return {
                "input_ids": torch.tensor(
                    ids, dtype=torch.long
                ),
                "labels": torch.tensor(
                    [-100] * len(p) + a + [EOS],
                    dtype=torch.long
                ),
            }, completion, len(ids)

    raise ValueError(
        "Scene cannot fit within 1024 tokens "
        "while retaining a long continuation."
    )

def load_pairs(split):
    grouped = defaultdict(list)

    with (
        SOURCE / f"{split}_cases.jsonl"
    ).open(encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            grouped[case["pair_id"]].append(case)

    pairs = []

    for pair in grouped.values():
        assert len(pair) == 2
        pair.sort(key=lambda c: c["variant"])
        assert pair[0]["characters"] == pair[1]["characters"]
        pairs.append(pair)

    return pairs

for split, seed, copies in (
    ("train", 1717, 2),
    ("val", 2727, 1),
):
    rng = random.Random(seed)
    buckets = {
        "character": [],
        "reasoning": [],
    }

    metadata = []
    lengths = []
    word_counts = []

    for pair_index, pair in enumerate(load_pairs(split)):
        for augmentation in range(copies):
            roles = role_mapping(pair, rng)
            task = (
                "character"
                if (pair_index + augmentation) % 2 == 0
                else "reasoning"
            )

            for case in pair:
                context = re.sub(
                    r"^(Story so far:|Events established in the story:)\s*",
                    "",
                    case["context"].strip()
                )

                replaced_context = replace_names(
                    context, roles
                )

                prompt = (
                    "Continue the novel scene using the established "
                    "character roles. Keep events and identities "
                    "consistent. Write descriptive narrative prose.\n\n"
                    "Previous scene:\n"
                    + replaced_context
                    + "\n\nNext scene:\n"
                )

                paragraphs = long_scene(case, roles, rng)
                record, completion, length = encode_record(
                    prompt, paragraphs
                )

                for original_name in roles:
                    assert not re.search(
                        rf"\b{re.escape(original_name)}\b",
                        prompt + completion
                    )

                assert "<CHAR_" in prompt
                assert "<CHAR_" in completion

                buckets[task].append(record)
                lengths.append(length)
                word_counts.append(len(completion.split()))

                metadata.append({
                    "pair_id": str(case["pair_id"]),
                    "variant": case["variant"],
                    "task": task,
                    "roles": roles,
                    "expected": case["expected"],
                    "prompt": prompt,
                    "completion": completion,
                })

    for task, records in buckets.items():
        if split == "train":
            rng.shuffle(records)

        folder = OUT / task
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(
            records,
            folder / f"{split}_sft.pt"
        )

        print(
            split, task, "records:", len(records)
        )

    with (
        OUT / f"{split}_examples.jsonl"
    ).open("w", encoding="utf-8") as f:
        for item in metadata:
            f.write(
                json.dumps(item, ensure_ascii=False)
                + "\n"
            )

    print(
        split.upper(),
        "minimum words:", min(word_counts),
        "average words:", round(
            sum(word_counts) / len(word_counts), 1
        ),
        "maximum tokens:", max(lengths)
    )

print("\nREV6 PLACEHOLDER LONG-SCENE BUILD: PASS")
