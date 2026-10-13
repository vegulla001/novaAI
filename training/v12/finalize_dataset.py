"""StoryAI V12: one safe, repeatable dataset finalization command.

No external models, downloads, or GPU required. Never silently promotes
machine-generated labels to human-approved status.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = ('story', 'source_book_id', 'title')

def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}')
    with path.open('r', encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number}: expected JSON object')
            yield value

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

def story_key(row):
    return hashlib.sha256(re.sub(r'\s+', ' ', row['story']).strip().encode('utf-8')).hexdigest()

def words(text):
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)

def valid(row, require_labels=False):
    issues = []
    for field in REQUIRED:
        if not str(row.get(field, '')).strip():
            issues.append('missing_' + field)
    story = row.get('story', '')
    if not isinstance(story, str) or len(words(story)) < 75:
        issues.append('too_short')
    if '\ufffd' in story or 'â€œ' in story or 'â€' in story:
        issues.append('encoding_problem')
    if re.search(r'(?im)^\s*(?:contents|table of contents|project gutenberg|copyright notice)\s*$', story):
        issues.append('front_matter')
    if require_labels:
        for field in ('objective', 'premise', 'setting'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                issues.append('missing_' + field)
        if not isinstance(row.get('characters'), list):
            issues.append('invalid_characters')
        else:
            for character in row['characters']:
                if not isinstance(character, str) or not re.search(r'(?<!\w)' + re.escape(character) + r'(?!\w)', story, re.I):
                    issues.append('ungrounded_character')
                    break
    return issues

def choose_books(book_counts, seed):
    """Group-disjoint, deterministic 3-way split. No passage leakage."""
    books = sorted(book_counts)
    if len(books) < 3:
        raise ValueError('At least 3 distinct source books are required for train/val/test.')
    random.Random(seed).shuffle(books)
    # Reserve two books for evaluation, then allocate remaining to training.
    # For six books, hold out one book each for val and test.
    return {book: ('val' if index == 0 else 'test' if index == 1 else 'train')
            for index, book in enumerate(books)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='dataset/v12/review_cleaned.jsonl')
    parser.add_argument('--approved', default='dataset/v12/curated_verified.jsonl')
    parser.add_argument('--output-dir', default='dataset/v12/final_v12')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, approved_path, out = Path(args.input), Path(args.approved), Path(args.output_dir)
    originals = {}
    rejected = []
    for row in read_jsonl(source):
        problems = valid(row)
        if problems:
            rejected.append({'source_book_id': row.get('source_book_id'), 'reason': problems,
                             'story_preview': str(row.get('story', ''))[:140]})
            continue
        key = story_key(row)
        if key in originals:
            rejected.append({'source_book_id': row.get('source_book_id'), 'reason': ['duplicate'], 'story_preview': row['story'][:140]})
            continue
        originals[key] = row
    book_counts = Counter(str(r['source_book_id']) for r in originals.values())
    mapping = choose_books(book_counts, args.seed)
    approved_by_key = {}
    for row in read_jsonl(approved_path):
        problems = valid(row, require_labels=True)
        if problems:
            raise ValueError('Previously approved record failed structural checks: ' + ', '.join(problems))
        key = story_key(row)
        if key not in originals:
            raise ValueError('Approved story is not present in cleaned source dataset.')
        if str(row['source_book_id']) != str(originals[key]['source_book_id']):
            raise ValueError('Approved record source book does not match source dataset.')
        approved_by_key[key] = row
    rows = defaultdict(list)
    approved_counts = Counter()
    for key, original in originals.items():
        split = mapping[str(original['source_book_id'])]
        if key in approved_by_key:
            row = dict(approved_by_key[key])
            row['review_status'] = 'HUMAN_APPROVED'
            approved_counts[split] += 1
        else:
            row = dict(original)
            row['review_status'] = 'NEEDS_REVIEW'
            # Do not preserve potentially inaccurate machine-generated labels.
            row['objective'] = ''
            row['premise'] = ''
        row['example_id'] = key
        rows[split].append(row)
    for split in ('train', 'val', 'test'):
        write_jsonl(out / f'{split}_candidates.jsonl', rows[split])
        write_jsonl(out / f'{split}_approved.jsonl', [r for r in rows[split] if r['review_status'] == 'HUMAN_APPROVED'])
    write_jsonl(out / 'excluded.jsonl', rejected)
    report = {
        'input': str(source), 'approved_input': str(approved_path),
        'unique_usable': len(originals), 'excluded': len(rejected),
        'split_sizes': {s: len(rows[s]) for s in ('train', 'val', 'test')},
        'approved_by_split': dict(approved_counts),
        'source_books_by_split': {s: sorted(b for b, group in mapping.items() if group == s)
                                  for s in ('train', 'val', 'test')},
        'ready_for_training': False,
        'reason': 'Unreviewed passages are NOT verified instruction examples. '
                  'Approved-only training data is insufficient for a reliable V12 fine-tune.',
        'safety': 'This command does not modify original datasets, code, or checkpoints.'
    }
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
