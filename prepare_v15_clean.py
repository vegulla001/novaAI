#!/usr/bin/env python3
"""Conservative, non-destructive V15 corpus preparation.

Requires: pip install numpy tokenizers
Example:
 python prepare_v15_clean.py --input-dir dataset/v12/prose_processed --tokenizer dataset/v15/tokenizer.json --out-dir dataset/v15_clean_candidate

Never modifies existing tokenizer, checkpoints or dataset/v15 binaries.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

START = re.compile(r'^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b.*$', re.I | re.M)
END = re.compile(r'^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK\b.*$', re.I | re.M)
TOC_LINE = re.compile(r'^\s*(?:chapter|book|part)\s+(?:[ivxlcdm]+|\d+)\s*[.:-]?\s+.{0,90}$', re.I)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def clean(text):
    text = text.replace('\ufeff', '').replace('\r\n', '\n').replace('\r', '\n')
    start, end = START.search(text), END.search(text)
    if start:
        text = text[start.end():]
    if end:
        # End offset is valid even after start removal only when re-searched.
        end2 = END.search(text)
        if end2:
            text = text[:end2.start()]
    text = re.sub(r'\n{4,}', '\n\n\n', text).strip()
    # Avoid deleting legitimate chapter headings; only remove obvious initial contents block.
    lines = text.splitlines()
    first = next((i for i, line in enumerate(lines[:100]) if line.strip().lower() in {'contents', 'table of contents'}), None)
    if first is not None:
        j = first + 1
        while j < min(len(lines), first + 90) and (not lines[j].strip() or TOC_LINE.match(lines[j])):
            j += 1
        if j - first >= 5:
            del lines[first:j]
            text = '\n'.join(lines).strip()
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='dataset/v12/prose_processed')
    p.add_argument('--tokenizer', default='dataset/v15/tokenizer.json')
    p.add_argument('--out-dir', default='dataset/v15_clean_candidate')
    p.add_argument('--min-chars', type=int, default=3000)
    p.add_argument('--val-ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-files', type=int, default=0, help='0 means all files')
    a = p.parse_args()
    if not 0 < a.val_ratio < 1:
        p.error('--val-ratio must be between 0 and 1')
    root, out, tokpath = Path(a.input_dir), Path(a.out_dir), Path(a.tokenizer)
    if not root.is_dir():
        p.error(f'Input directory not found: {root}')
    if not tokpath.is_file():
        p.error(f'Tokenizer not found: {tokpath}')
    if out.resolve() == Path('dataset/v15').resolve():
        p.error('Refusing to overwrite dataset/v15. Choose a candidate output directory.')
    if out.exists() and any(out.iterdir()):
        p.error(f'Output directory is not empty: {out}. Choose a new directory.')
    tok = Tokenizer.from_file(str(tokpath))
    vocab = tok.get_vocab_size()
    bos, eos = tok.token_to_id('<bos>'), tok.token_to_id('<eos>')
    if bos is None or eos is None:
        p.error('Existing tokenizer does not contain <bos> and <eos>. Do not proceed until verified.')
    paths = sorted(root.rglob('*.txt'))
    if a.max_files:
        paths = paths[:a.max_files]
    if not paths:
        p.error(f'No .txt files under {root}')
    docs, seen, skipped = [], set(), []
    for path in paths:
        text = clean(path.read_text(encoding='utf-8-sig', errors='replace'))
        if '\ufffd' in text:
            skipped.append({'file': str(path), 'reason': 'replacement_characters'})
            continue
        if len(text) < a.min_chars:
            skipped.append({'file': str(path), 'reason': 'too_short', 'chars': len(text)})
            continue
        fingerprint = hashlib.sha256(re.sub(r'\s+', ' ', text).strip().casefold().encode('utf-8')).hexdigest()
        if fingerprint in seen:
            skipped.append({'file': str(path), 'reason': 'exact_duplicate'})
            continue
        seen.add(fingerprint)
        docs.append((path, text))
    if len(docs) < 3:
        p.error(f'Only {len(docs)} usable documents. Need at least 3.')
    random.Random(a.seed).shuffle(docs)
    nval = max(1, round(len(docs) * a.val_ratio))
    splits = {'val': docs[:nval], 'train': docs[nval:]}
    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    report = {'source_dir': str(root), 'tokenizer_path': str(tokpath), 'tokenizer_sha256': sha256(tokpath),
              'vocab_size': vocab, 'bos_id': bos, 'eos_id': eos, 'seed': a.seed,
              'cleaning': 'Gutenberg markers, obvious initial contents lists, whitespace; no heuristic prose rewriting',
              'skipped': skipped, 'splits': {}}
    dtype = np.uint16 if vocab <= 65536 else np.uint32
    for split, items in splits.items():
        arrays, entries, total = [], [], 0
        for path, story in items:
            ids = tok.encode(story, add_special_tokens=False).ids
            if not ids:
                skipped.append({'file': str(path), 'reason': 'empty_tokenization'})
                continue
            if max(ids) >= vocab:
                raise ValueError(f'Out-of-range token in {path}')
            # Explicit boundaries, consistent with decoded existing V15 corpus.
            arr = np.asarray([bos] + ids + [eos], dtype=dtype)
            arrays.append(arr)
            entries.append({'file': str(path), 'tokens': len(arr), 'chars': len(story)})
            total += len(arr)
        if not arrays:
            raise ValueError(f'No encoded documents in {split}')
        target = out / f'{split}.npy'
        np.save(target, np.concatenate(arrays))
        report['splits'][split] = {'tokens': total, 'documents': entries, 'sha256': sha256(target)}
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(out), 'tokenizer_sha256': report['tokenizer_sha256'],
                      'train_docs': len(report['splits']['train']['documents']),
                      'val_docs': len(report['splits']['val']['documents']),
                      'train_tokens': report['splits']['train']['tokens'],
                      'val_tokens': report['splits']['val']['tokens'],
                      'skipped_docs': len(skipped)}, indent=2))
    print('Candidate only. Do not train yet; review corpus and validate token distribution first.')

if __name__ == '__main__':
    main()
