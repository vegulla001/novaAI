import torch

def instruction_for(record):
    characters = ", ".join(record["characters"])

    return (
        "<INSTRUCTION>\n"
        f"Characters: {characters}\n"
        f"Setting: {record['setting']}\n"
        f"Goal: {record['objective']}\n"
        f"Problem: {record['obstacle']}\n"
        f"End: {record['outcome']}\n"
    )


def plan_for(record):
    characters = ", ".join(record["characters"])

    return (
        "<PLAN>\n"
        f"Characters: {characters}\n"
        f"Goal: {record['objective']}\n"
        f"Problem: {record['obstacle']}\n"
        f"End: {record['outcome']}\n"
    )

def build_example(record, tokenizer, block_size):
    characters = ", ".join(record["characters"])

    # Compact instruction: preserve the important facts.
    instruction = instruction_for(record)

    # The model learns to reproduce these requirements.
    plan = plan_for(record)

    story_prefix = "<STORY>\n"
    story_text = record["story"].strip()
    story_suffix = "\n<END_STORY>"

    instruction_ids = tokenizer.encode(instruction)
    plan_ids = tokenizer.encode(plan)
    prefix_ids = tokenizer.encode(story_prefix)
    story_ids = tokenizer.encode(story_text)
    suffix_ids = tokenizer.encode(story_suffix)

    # Preserve the instruction, plan and story markers.
    fixed_length = (
        len(instruction_ids)
        + len(plan_ids)
        + len(prefix_ids)
        + len(suffix_ids)
    )

    # x has at most block_size tokens; targets are shifted.
    available_story_tokens = block_size + 1 - fixed_length

    if available_story_tokens < 1:
        raise ValueError(
            f"{record['title']}: instruction and plan "
            f"are too long for context size {block_size}"
        )

    was_truncated = len(story_ids) > available_story_tokens

    if was_truncated:
        # Preserve the beginning of the story.
        story_ids = story_ids[:available_story_tokens]

        # Do not pretend a truncated story has a real ending.
        suffix_ids = []

    full_ids = (
        instruction_ids
        + plan_ids
        + prefix_ids
        + story_ids
        + suffix_ids
    )

    x = torch.tensor(
        full_ids[:-1],
        dtype=torch.long
    ).unsqueeze(0)

    next_tokens = torch.tensor(
        full_ids[1:],
        dtype=torch.long
    ).unsqueeze(0)

    # Plan loss: predict plan tokens only.
    plan_targets = next_tokens.clone()

    plan_start = len(instruction_ids) - 1
    plan_end = len(instruction_ids) + len(plan_ids) - 1

    plan_targets[:, :plan_start] = -1
    plan_targets[:, plan_end:] = -1

    # Story loss: predict story tokens only.
    story_targets = next_tokens.clone()

    story_start = (
        len(instruction_ids)
        + len(plan_ids)
        - 1
    )

    story_targets[:, :story_start] = -1

    assert x.shape[1] <= block_size
    assert (plan_targets != -1).any()
    assert (story_targets != -1).any()

    return x, plan_targets, story_targets