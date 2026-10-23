#!/usr/bin/env python3
"""Build V15 pretraining token streams, reusing a fixed tokenizer by default."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>', '<story>', '<chapter>', '<scene>', '<end_scene>']


def records(path):
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = obj.get('text')
            if not isinstance(txt, str) or not txt.strip():
                raise ValueError(f'{path}:{line_no}: missing nonempty text')
            if 'book_id' not in obj:
                raise ValueError(f'{path}:{line_no}: missing book_id')
            yield obj


def texts(path):
    for r in records(path):
        yield r['text']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='dataset/v15/prose/train.jsonl')
    p.add_argument('--val', default='dataset/v15/prose/val.jsonl')
    p.add_argument('--outdir', default='dataset/v15/tokenizer')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--existing-tokenizer', help='Reuse this tokenizer JSON without modifying it')
    mode.add_argument('--train-new-tokenizer', action='store_true', help='Explicitly train a new BPE tokenizer')
    p.add_argument('--vocab-size', type=int, default=8192)
    p.add_argument('--expected-sha256', help='Required tokenizer SHA-256, recommended for fixed-tokenizer builds')
    args = p.parse_args()

    train, val = Path(args.train), Path(args.val)
    if not train.is_file() or not val.is_file():
        p.error('Missing prepared train/val JSONL.')
    train_ids = {str(r['book_id']) for r in records(train)}
    val_ids = {str(r['book_id']) for r in records(val)}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError('Train/validation book_id overlap: ' + str(sorted(overlap)))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.existing_tokenizer:
        tok_path = Path(args.existing_tokenizer).resolve()
        if not tok_path.is_file():
            p.error(f'Missing tokenizer: {tok_path}')
        if args.expected_sha256 and sha(tok_path).lower() != args.expected_sha256.lower():
            raise RuntimeError('Tokenizer SHA-256 mismatch; refusing to encode')
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        tok_path = out / 'tokenizer.json'
        if tok_path.exists():
            raise RuntimeError(f'Refusing to overwrite existing tokenizer: {tok_path}')
        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(texts(train), trainer=trainer)
        tokenizer.save(str(tok_path))
        if args.expected_sha256 and sha(tok_path).lower() != args.expected_sha256.lower():
            raise RuntimeError('New tokenizer SHA-256 does not match expected hash')

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != args.vocab_size:
        raise RuntimeError(f'Actual vocab {vocab_size} != expected {args.vocab_size}')
    special_ids = {s: tokenizer.token_to_id(s) for s in SPECIAL}
    if any(v is None for v in special_ids.values()) or len(set(special_ids.values())) != len(SPECIAL):
        raise RuntimeError(f'Missing or duplicate special token IDs: {special_ids}')
    bos, eos = special_ids['<bos>'], special_ids['<eos>']
    if vocab_size > 65536:
        raise RuntimeError('uint16 token streams cannot represent this vocabulary')

    sample = 'Rhea opened the archive door. Pax waited in the corridor.'
    decoded = tokenizer.decode(tokenizer.encode(sample, add_special_tokens=False).ids)
    if decoded != sample:
        raise RuntimeError(f'Round-trip mismatch: {decoded!r}')

    report = {
        'vocab_size': vocab_size,
        'tokenizer_path': str(tok_path),
        'tokenizer_sha256': sha(tok_path),
        'train_book_count': len(train_ids),
        'val_book_count': len(val_ids),
        'book_overlap': 0,
        'special_tokens': special_ids,
        'splits': {},
    }
    for split, path in [('train', train), ('val', val)]:
        arr = []
        count = 0
        for r in records(path):
            arr.append(bos)
            arr.extend(tokenizer.encode(r['text'], add_special_tokens=False).ids)
            arr.append(eos)
            count += 1
        if not arr or min(arr) < 0 or max(arr) >= vocab_size:
            raise RuntimeError(f'Invalid token stream for {split}')
        target = out / f'{split}.npy'
        # Write to a temporary file first; avoid partially-written .npy on failure.
        temporary = out / f'{split}.npy.tmp'
        with temporary.open('wb') as f:
            np.save(f, np.asarray(arr, dtype=np.uint16))
        temporary.replace(target)
        report['splits'][split] = {
            'records': count, 'tokens': len(arr), 'file': str(target), 'sha256': sha(target)
        }
        print(f'{split}: {count} records, {len(arr):,} tokens -> {target}')

    (out / 'build_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print('PASS: book-disjoint splits, tokenizer round-trip, token ID bounds')
    print('Tokenizer SHA256:', report['tokenizer_sha256'])
    print('NOTE: these are pretraining token streams, NOT instruction fine-tuning examples.')


if __name__ == '__main__':
    main()
