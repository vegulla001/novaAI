#!/usr/bin/env python3
"""Offline, provenance-first V15 corpus expansion. Standard library only."""
import argparse, csv, hashlib, json, re, unicodedata
from collections import Counter
from pathlib import Path

FIELDS = ('source_id', 'path', 'title', 'author', 'license', 'license_evidence', 'approved', 'split')
LICENSES = {'public-domain-verified', 'cc0', 'cc-by-4.0', 'cc-by-sa-4.0', 'owned-rights'}

def normalize(s):
    s = unicodedata.normalize('NFC', s).replace('\ufeff', '')
    s = re.sub(r'\[Illustration[^\]]*\]', '', s, flags=re.I)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def digest(s):
    return hashlib.sha256(re.sub(r'\s+', ' ', s).casefold().encode('utf-8')).hexdigest()

def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{n}: {e}') from e

def iter_source(path):
    if path.suffix.lower() == '.jsonl':
        for row in read_jsonl(path):
            if not isinstance(row.get('text'), str): raise ValueError(f'Missing text in {path}')
            yield row['text']
    elif path.suffix.lower() == '.txt':
        text = path.read_text(encoding='utf-8-sig')
        # Split into moderately sized paragraphs without inventing metadata.
        for block in re.split(r'\n\s*\n', text):
            for i in range(0, len(block), 5000):
                yield block[i:i+5000]
    else: raise ValueError(f'Unsupported source extension: {path}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--existing-train', default='dataset/v15/prose/train.jsonl')
    p.add_argument('--existing-val', default='dataset/v15/prose/val.jsonl')
    p.add_argument('--outdir', default='dataset/v15/expansion')
    p.add_argument('--min-chars', type=int, default=180)
    args = p.parse_args()
    manifest = Path(args.manifest).resolve()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    existing_ids = set(); seen = set(); counts = Counter(); sources = []
    for split, file in [('train', args.existing_train), ('val', args.existing_val)]:
        path = Path(file)
        if not path.is_file(): raise SystemExit(f'Missing existing {split} file: {path}')
        for row in read_jsonl(path):
            book_id = str(row.get('book_id', ''))
            if not book_id: raise SystemExit(f'Missing book_id in {path}')
            existing_ids.add(book_id)
            if row.get('text'): seen.add(digest(normalize(row['text'])))
    new_ids = set()
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise SystemExit(f'Manifest must have columns: {", ".join(FIELDS)}')
        rows = list(reader)
    for row in rows:
        sid = row['source_id'].strip()
        if not sid or sid in existing_ids or sid in new_ids:
            raise SystemExit(f'Duplicate or colliding source_id: {sid!r}')
        new_ids.add(sid)
    with (out/'new_train.jsonl').open('w', encoding='utf-8') as train, (out/'new_val.jsonl').open('w', encoding='utf-8') as val:
        for row in rows:
            sid = row['source_id'].strip()
            record = {'source_id': sid, 'title': row['title'], 'author': row['author'], 'license': row['license'], 'license_evidence': row['license_evidence'], 'split': row['split']}
            if row['approved'].strip().lower() != 'yes':
                record['status'] = 'rejected_unapproved'; sources.append(record); continue
            if row['license'].strip().lower() not in LICENSES or not row['license_evidence'].strip():
                record['status'] = 'rejected_license_not_documented'; sources.append(record); continue
            if row['split'].strip().lower() not in ('train', 'val'):
                record['status'] = 'rejected_invalid_split'; sources.append(record); continue
            path = (manifest.parent / row['path']).resolve()
            if not path.is_file():
                record['status'] = 'rejected_missing_file'; sources.append(record); continue
            target = train if row['split'].strip().lower() == 'train' else val
            accepted = 0
            for raw in iter_source(path):
                s = normalize(raw)
                if len(s) < args.min_chars: counts['too_short'] += 1; continue
                h = digest(s)
                if h in seen: counts['exact_duplicate'] += 1; continue
                seen.add(h)
                obj = {'text': s, 'book_id': sid, 'title': row['title'], 'author': row['author'], 'source_id': sid, 'license': row['license'], 'license_evidence': row['license_evidence'], 'type': 'prose_not_instruction'}
                target.write(json.dumps(obj, ensure_ascii=False)+'\n')
                accepted += 1; counts['new_'+row['split'].strip().lower()] += 1
                counts['new_characters'] += len(s)
            record['status'] = 'accepted' if accepted else 'no_usable_passages'
            record['segments'] = accepted
            sources.append(record)
    report = {'new_counts': dict(counts), 'sources': sources, 'existing_books': len(existing_ids), 'license_policy': 'Manifest approvals are assertions by the operator, not independent legal verification.', 'limitations': ['Exact normalized duplicate filtering only; near-duplicate review remains necessary.', 'Plain-text files are split by paragraphs and 5000-character slices; inspect quality.', 'Existing train/val files are not modified.', 'New split assignment must be done by source/book, never by segment.']}
    (out/'expansion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'new_counts': dict(counts), 'report': str(out/'expansion_report.json')}, indent=2))
    print('Review report and samples before merging. Existing data is unchanged.')

if __name__ == '__main__': main()
