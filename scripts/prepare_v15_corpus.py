#!/usr/bin/env python3
"""Safe incremental V15 corpus import. Dry-run by default; --apply commits."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = {'book_id','title','file_path','source_url','license','approved','split'}
TRUE = {'true','yes','1'}
FALSE = {'false','no','0',''}

def normalized_title(s):
    return re.sub(r'[^a-z0-9]', '', s.casefold())

def clean_text(s):
    s = s.replace('\ufeff', '').replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def split_text(text, target=2600):
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

def read_jsonl(path):
    if not path.is_file():
        raise RuntimeError(f'Missing {path}')
    records = []
    with path.open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f'{path}:{line_no}: {e}') from e
            if not isinstance(rec, dict) or not isinstance(rec.get('text'), str) or not rec['text'].strip() or not rec.get('book_id'):
                raise RuntimeError(f'{path}:{line_no}: invalid text/book_id')
            records.append(rec)
    return records

def write_jsonl(path, rows):
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', default='dataset/v15')
    ap.add_argument('--apply', action='store_true', help='Back up and write changes (default: dry-run)')
    ap.add_argument('--book-id', action='append', help='Restrict to named book ID (repeatable)')
    args = ap.parse_args()
    base = Path(args.base)
    prose = base / 'prose'
    sources = base / 'sources'
    manifest = sources / 'manifest.csv'
    train_path, val_path = prose / 'train.jsonl', prose / 'val.jsonl'
    train, val = read_jsonl(train_path), read_jsonl(val_path)
    if not manifest.is_file():
        raise RuntimeError(f'Missing {manifest}')
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise RuntimeError(f'Manifest requires {sorted(REQUIRED)}')
        manifest_rows = list(reader)
    book_splits, title_index, seen_text = {}, {}, set()
    for split, rows in [('train', train), ('val', val)]:
        for rec in rows:
            bid = str(rec['book_id'])
            if bid in book_splits and book_splits[bid] != split:
                raise RuntimeError(f'Book {bid} occurs in both splits')
            book_splits[bid] = split
            title = str(rec.get('title') or rec.get('book_title') or '')
            if title:
                title_index.setdefault(normalized_title(title), set()).add(bid)
            seen_text.add(hashlib.sha256(clean_text(rec['text']).casefold().encode()).hexdigest())
    existing_ids = set(book_splits)
    added = Counter()
    skipped, blocked = [], []
    seen_manifest_ids = set()
    selected = set(args.book_id or [])
    for row_no, row in enumerate(manifest_rows, 2):
        bid = (row.get('book_id') or '').strip()
        if not bid:
            blocked.append(f'row {row_no}: empty book_id')
            continue
        if bid in seen_manifest_ids:
            blocked.append(f'row {row_no}: repeated manifest book_id {bid}')
            continue
        seen_manifest_ids.add(bid)
        if selected and bid not in selected:
            continue
        approved = (row.get('approved') or '').strip().lower()
        if approved not in TRUE | FALSE:
            blocked.append(f'{bid}: invalid approved value {approved!r}')
            continue
        if approved not in TRUE:
            skipped.append(f'{bid}: unapproved')
            continue
        if bid in existing_ids:
            skipped.append(f'{bid}: already imported')
            continue
        title = (row.get('title') or '').strip()
        split = (row.get('split') or '').strip().lower()
        license_status = (row.get('license') or '').strip().lower()
        source_url = (row.get('source_url') or '').strip()
        if license_status != 'verified':
            blocked.append(f'{bid}: license is not verified')
            continue
        if not title or not source_url or split not in {'train', 'val'}:
            blocked.append(f'{bid}: missing title/source URL or invalid split')
            continue
        same_title = title_index.get(normalized_title(title), set()) - {bid}
        if same_title:
            blocked.append(f'{bid}: possible duplicate title already in corpus under {sorted(same_title)}')
            continue
        rel = Path((row.get('file_path') or '').strip())
        if not rel.parts or rel.is_absolute() or '..' in rel.parts or rel.suffix.lower() != '.txt':
            blocked.append(f'{bid}: invalid source-relative txt path')
            continue
        path = (sources / rel).resolve()
        if not path.is_relative_to(sources.resolve()) or not path.is_file():
            blocked.append(f'{bid}: missing/unsafe file {rel}')
            continue
        raw = path.read_text(encoding='utf-8-sig')
        if re.search(r'\*\*\*\s*(START|END) OF (THE|THIS) PROJECT GUTENBERG', raw, re.I):
            blocked.append(f'{bid}: Gutenberg boilerplate markers remain')
            continue
        chunks = split_text(clean_text(raw))
        if not chunks:
            blocked.append(f'{bid}: no usable chunks')
            continue
        staged = []
        for chunk in chunks:
            digest = hashlib.sha256(chunk.casefold().encode()).hexdigest()
            if digest in seen_text:
                continue
            staged.append((chunk, digest))
        if not staged:
            blocked.append(f'{bid}: all chunks duplicated')
            continue
        dest = train if split == 'train' else val
        for chunk, digest in staged:
            seen_text.add(digest)
            dest.append({'text': chunk, 'book_id': bid, 'title': title,
                         'genre': 'unspecified', 'source_url': source_url,
                         'type': 'licensed_prose_not_instruction'})
        added[split] += len(staged)
        book_splits[bid] = split
        title_index.setdefault(normalized_title(title), set()).add(bid)
        print(f'READY {bid}: {len(staged)} segments -> {split}')
    report = {'mode': 'apply' if args.apply else 'dry-run',
              'train_records': len(train), 'val_records': len(val),
              'train_books': sum(s == 'train' for s in book_splits.values()),
              'val_books': sum(s == 'val' for s in book_splits.values()),
              'added_segments': dict(added), 'skipped': skipped, 'blocked': blocked,
              'note': 'License=verified must be independently reviewed; title and exact-segment checks do not catch all edition/near duplicates.'}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if blocked:
        print('BLOCKED: Resolve flagged entries before applying. No files changed.')
        return 2
    if not args.apply:
        print('DRY RUN: no files changed. Use --apply after review.')
        return 0
    if not added:
        print('No new segments; files unchanged.')
        return 0
    backup = prose / 'backups' / datetime.now(timezone.utc).strftime('pre_import_%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True, exist_ok=False)
    for path in (train_path, val_path, manifest):
        shutil.copy2(path, backup / path.name)
    print(f'Backup: {backup}')
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    (prose / 'import_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('APPLIED: audit corpus, then rebuild .npy using existing tokenizer.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
