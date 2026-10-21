#!/usr/bin/env python3
"""Incrementally import approved .txt books into dataset/v15/prose.

Existing prose records are preserved. Each new book is imported once by book_id.
Run from the project root: python scripts/prepare_v15_corpus.py
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def clean_text(s):
    s = s.replace('\ufeff', '').replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def split_text(text, target=2600):
    # Split at paragraph boundaries when possible, retaining the full text.
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current = [], ''
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 2 <= target:
            current = f'{current}\n\n{p}' if current else p
        else:
            if current:
                chunks.append(current)
                current = ''
            while len(p) > target:
                cut = p.rfind(' ', 0, target)
                if cut < target // 2:
                    cut = target
                chunks.append(p[:cut].strip())
                p = p[cut:].strip()
            current = p
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 80]


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open('r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc
                if not isinstance(rec.get('text'), str) or not rec.get('book_id'):
                    raise ValueError(f'{path}:{line_no}: missing text/book_id')
                records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='dataset/v15')
    args = ap.parse_args()
    base = Path(args.base)
    prose = base / 'prose'
    sources = base / 'sources'
    manifest = sources / 'manifest.csv'
    if not manifest.exists():
        raise SystemExit(f'Missing {manifest}; create it with headers: book_id,title,file_path,source_url,license,approved,split')
    train_path, val_path = prose / 'train.jsonl', prose / 'val.jsonl'
    if not train_path.exists() or not val_path.exists():
        raise SystemExit('Existing prose/train.jsonl and prose/val.jsonl required. No files changed.')

    train, val = load_jsonl(train_path), load_jsonl(val_path)
    book_splits = {}
    seen_text = set()
    for split, rows in [('train', train), ('val', val)]:
        for rec in rows:
            bid = str(rec['book_id'])
            if bid in book_splits and book_splits[bid] != split:
                raise SystemExit(f'Book ID {bid} appears in both splits. No files changed.')
            book_splits[bid] = split
            seen_text.add(hashlib.sha256(clean_text(rec['text']).casefold().encode('utf-8')).hexdigest())

    required = {'book_id','title','file_path','source_url','license','approved','split'}
    added = Counter()
    skipped = []
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f'Manifest must have columns: {", ".join(sorted(required))}')
        for row_no, row in enumerate(reader, 2):
            if not any((v or '').strip() for v in row.values()):
                continue
            bid = (row.get('book_id') or '').strip()
            title = (row.get('title') or '').strip()
            approved = (row.get('approved') or '').strip().lower() in {'true','yes','1'}
            if not approved:
                skipped.append(f'row {row_no}: not approved')
                continue
            if not bid or not title or not (row.get('license') or '').strip() or not (row.get('source_url') or '').strip():
                raise SystemExit(f'Manifest row {row_no}: approved books require id/title/license/source_url')
            split = (row.get('split') or '').strip().lower()
            if split not in {'train','val'}:
                raise SystemExit(f'Manifest row {row_no}: split must be train or val')
            if bid in book_splits:
                if book_splits[bid] != split:
                    raise SystemExit(f'Manifest row {row_no}: book {bid} already in {book_splits[bid]}, requested {split}')
                skipped.append(f'row {row_no}: book {bid} already imported')
                continue
            rel = Path((row.get('file_path') or '').strip())
            if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.suffix.lower() != '.txt':
                raise SystemExit(f'Manifest row {row_no}: file_path must be a relative .txt path within sources/')
            path = (sources / rel).resolve()
            if not path.is_relative_to(sources.resolve()) or not path.is_file():
                raise SystemExit(f'Manifest row {row_no}: source file not found inside sources/: {rel}')
            raw = path.read_text(encoding='utf-8-sig')
            # This is conservative whitespace cleanup, NOT reliable Gutenberg header removal.
            chunks = split_text(clean_text(raw))
            if not chunks:
                raise SystemExit(f'Manifest row {row_no}: no usable text in {path}')
            destination = train if split == 'train' else val
            for chunk in chunks:
                digest = hashlib.sha256(chunk.casefold().encode('utf-8')).hexdigest()
                if digest in seen_text:
                    continue
                seen_text.add(digest)
                destination.append({
                    'text': chunk, 'book_id': bid, 'title': title,
                    'genre': 'unspecified', 'source_url': row['source_url'].strip(),
                    'type': 'public_domain_prose_not_instruction' if 'public' in row['license'].lower() else 'licensed_prose_not_instruction'
                })
                added[split] += 1
            book_splits[bid] = split

    # Write all outputs to temporary files first, then replace existing JSONL.
    prose.mkdir(parents=True, exist_ok=True)
    for path, records in [(train_path, train), (val_path, val)]:
        tmp = path.with_suffix('.jsonl.tmp')
        with tmp.open('w', encoding='utf-8', newline='\n') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        tmp.replace(path)
    report = {
        'train_records': len(train), 'val_records': len(val),
        'train_books': sum(x == 'train' for x in book_splits.values()),
        'val_books': sum(x == 'val' for x in book_splits.values()),
        'added_segments': dict(added), 'skipped_manifest_rows': skipped,
        'note': 'Whitespace cleanup and exact segment dedup only; manually remove front matter and check near duplicates and licensing.'
    }
    (prose / 'import_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('NEXT: rebuild tokenizer/binaries only after inspecting imported prose.')

if __name__ == '__main__':
    main()
