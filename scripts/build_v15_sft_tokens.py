#!/usr/bin/env python3
"""Encode StoryAI V15 instruction/response JSONL for response-only SFT.

Run from the project root. Does not alter pretraining data or tokenizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import torch
from tokenizers import Tokenizer


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    with path.open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get('instruction'), str) or not record['instruction'].strip():
                raise ValueError(f'{path}:{number}: missing instruction')
            if not isinstance(record.get('response'), str) or not record['response'].strip():
                raise ValueError(f'{path}:{number}: missing response')
            result.append(record)
    if not result:
        raise ValueError(f'Empty split: {path}')
    return result

OUTLINE_PATTERNS = [
    r"(?im)^\s*Act\s+[IVX]+\s*[—:-]",
    r"(?im)^\s*\d+\.\s*(Goal|Obstacle|Tactic|Reversal|Choice|Hook)\s*:",
    r"(?im)^\s*(Title|Premise|Character change)\s*:",
]


def validate_narrative_example(row, uid):
    instruction = row["instruction"].strip()
    response = row["response"].strip()

    if len(response.split()) < 150:
        raise ValueError(
            f"{uid}: narrative response must contain at least 150 words"
        )

    for pattern in OUTLINE_PATTERNS:
        if re.search(pattern, response):
            raise ValueError(
                f"{uid}: outline-style response found in narrative dataset"
            )

    if not any(mark in response for mark in (".", "!", "?")):
        raise ValueError(
            f"{uid}: response does not contain sentence punctuation"
        )

    return instruction, response

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft-dir', default='dataset/v15/sft')
    ap.add_argument('--tokenizer', default='dataset/v15/tokenizer.json')
    ap.add_argument('--out-dir', default='dataset/v15/tokenizer/instructions')
    ap.add_argument('--expected-sha256', default='ab2daa146b2261c18bf4f8f3122136be30100a230e675ba7d70783a1fc29090d')
    ap.add_argument('--apply', action='store_true', help='Write train_sft.pt and val_sft.pt')
    args = ap.parse_args()

    tokenizer_path = Path(args.tokenizer)
    digest = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if digest != args.expected_sha256:
        raise ValueError(f'Tokenizer SHA256 mismatch: {digest}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos, eos = tokenizer.token_to_id('<bos>'), tokenizer.token_to_id('<eos>')
    if bos is None or eos is None:
        raise ValueError('Tokenizer is missing <bos> or <eos>')

    base = Path(args.sft_dir)
    out = Path(args.out_dir)
    all_ids = set()
    encoded = {}
    for split in ('train', 'val'):
        rows = read_jsonl(base / f'{split}_generated.jsonl')
        examples = []
        categories = {}
        for index, row in enumerate(rows, 1):
            uid = str(row.get('id') or f'{split}-{index}')
            if row.get("category") != "narrative":
                raise ValueError(
                    f"{uid}: expected category='narrative', "
                    f"got {row.get('category')!r}"
                )

            validate_narrative_example(row, uid)
            if uid in all_ids:
                raise ValueError(f'Duplicate example id across splits: {uid}')
            all_ids.add(uid)
            category = str(row.get('category', 'unspecified'))
            categories[category] = categories.get(category, 0) + 1
            prompt = f"Instruction:\n{row['instruction'].strip()}\n\nResponse:\n"
            completion = row['response'].strip()
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            answer_ids = tokenizer.encode(completion, add_special_tokens=False).ids
            if not prompt_ids or not answer_ids:
                raise ValueError(f'Empty tokenization for {uid}')
            input_ids = [bos] + prompt_ids + answer_ids + [eos]
            labels = [-100] * (1 + len(prompt_ids)) + answer_ids + [eos]
            assert len(input_ids) == len(labels)
            examples.append({'input_ids': input_ids, 'labels': labels})
        encoded[split] = examples
        lengths = [len(e['input_ids']) for e in examples]
        print(f'{split}: {len(examples)} examples, min={min(lengths)}, max={max(lengths)}, '
              f'over_4096={sum(n > 4097 for n in lengths)}, categories={categories}')

    if args.apply:
        out.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'val'):
            dest = out / f'{split}_sft.pt'
            if dest.exists():
                raise FileExistsError(f'{dest} already exists; move or back it up first')
            torch.save(encoded[split], dest)
            print('Wrote:', dest)
    else:
        print('DRY RUN: no files written. Add --apply to save tokenized SFT data.')


if __name__ == '__main__':
    main()
