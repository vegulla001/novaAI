"""Interactive, resumable human review of StoryAI V12 instruction/story candidates.

No third-party packages required. Run from the repository root:
  python -m training.v12.review_dataset --limit 20

Decisions are persisted in a separate JSONL log. Approved examples are exported
only after required fields and explicit character mentions have been checked.
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

DEFAULT_INPUT = 'dataset/v12/review_cleaned.jsonl'
DEFAULT_DECISIONS = 'dataset/v12/review_decisions.jsonl'
DEFAULT_APPROVED = 'dataset/v12/curated_verified.jsonl'


def read_jsonl(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Missing input file: {p}')
    for line_no, line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{p}:{line_no}: invalid JSON: {exc}') from exc
        if not isinstance(item, dict):
            raise ValueError(f'{p}:{line_no}: expected JSON object')
        yield item


def save_decision(path, item):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()


def export_approved(decisions, output):
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    approved = [d['example'] for _, d in sorted(decisions.items()) if d.get('status') == 'APPROVED']
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        for example in approved:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    temp.replace(p)
    return len(approved)


def ask(label, default=''):
    result = input(f'{label} [{default}]: ').strip()
    return result if result else default


def normalize_names(raw):
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).split(',') if s.strip()]


def validate(example):
    errors = []
    story = example.get('story', '').strip()
    if len(story.split()) < 35:
        errors.append('Story too short (<35 words).')
    for field in ('setting', 'objective', 'premise'):
        if not str(example.get(field, '')).strip():
            errors.append(f'Missing {field}.')
    names = normalize_names(example.get('characters', []))
    if not names:
        errors.append('At least one explicitly named character required.')
    for name in names:
        if not re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', story, flags=re.IGNORECASE):
            errors.append(f'Character not explicitly mentioned in passage: {name!r}')
    if not example.get('genres'):
        errors.append('Missing genre.')
    return errors


def draft_labels(row):
    """Conservative extractive suggestions; never mark them verified."""
    candidate = dict(row)
    story = re.sub(r"\s+", " ", str(row.get('story', '')).strip())
    names = normalize_names(row.get('characters', []))
    # Prefer an early sentence explicitly naming a suggested character.
    sentences = re.split(r'(?<=[.!?])\s+(?=[“"A-Z])', story)
    sentences = [s.strip(' “"') for s in sentences if len(s.split()) >= 6]
    named = [s for s in sentences[:8] if any(re.search(r'(?<!\w)' + re.escape(n) + r'(?!\w)', s, re.I) for n in names)]
    evidence = (named or sentences or [story[:240]])[0].strip()
    # A short extractive premise avoids inventing actions not in the passage.
    evidence = evidence[:300].rsplit(' ', 1)[0] + '…' if len(evidence) > 300 else evidence
    if not str(candidate.get('premise', '')).strip():
        candidate['premise'] = evidence
    if not str(candidate.get('objective', '')).strip():
        who = names[0] if names else 'the characters'
        candidate['objective'] = f'Depict the events involving {who} described in this passage: {evidence}'
    # No reliable automatic setting inference beyond the existing suggestion.
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--decisions', default=DEFAULT_DECISIONS)
    parser.add_argument('--output', default=DEFAULT_APPROVED)
    parser.add_argument('--limit', type=int, default=20, help='Maximum unreviewed examples in this session; 0 = unlimited')
    parser.add_argument('--show-words', type=int, default=250)
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--no-drafts', action='store_true', help='Disable automatic extractive label suggestions')
    parser.add_argument('--preview-drafts', action='store_true', help='Show draft labels without saving decisions')
    args = parser.parse_args()

    decisions = {}
    if Path(args.decisions).exists():
        for record in read_jsonl(args.decisions):
            decisions[int(record['index'])] = record  # latest decision wins

    if args.export_only:
        print(f'Exported {export_approved(decisions, args.output)} approved examples to {args.output}')
        return

    rows = list(read_jsonl(args.input))
    completed = 0
    for index, row in enumerate(rows):
        if index in decisions and decisions[index].get('status') in ('APPROVED', 'REJECTED'):
            continue
        if args.limit and completed >= args.limit:
            break
        print('\n' + '=' * 78)
        print(f'Candidate {index + 1}/{len(rows)} | book {row.get("source_book_id", "?")} | {row.get("title", "Untitled")}')
        print('=' * 78)
        words = str(row.get('story', '')).split()
        print(textwrap.fill(' '.join(words[:args.show_words]), width=90))
        if len(words) > args.show_words:
            print(f'\n[Preview truncated: {len(words)} words total. Review full source before approval.]')
        draft = dict(row) if args.no_drafts else draft_labels(row)
        print('\nSuggested characters:', draft.get('characters', []))
        print('Suggested setting:', draft.get('setting', ''))
        print('Draft objective:', draft.get('objective', ''))
        print('Draft premise:', draft.get('premise', ''))
        if not args.no_drafts:
            print('DRAFT ONLY: Extractive suggestions may be vague or wrong. Edit and verify before approval.')
        if args.preview_drafts:
            completed += 1
            continue
        print('\nChoose: [e]dit and approve, [r]eject, [s]kip, [v]iew full passage, [q]uit')
        while True:
            action = input('Action: ').strip().lower()
            if action == 'v':
                print('\n' + str(row.get('story', '')) + '\n')
                continue
            if action in ('e', 'r', 's', 'q'):
                break
            print('Choose e, r, s, v or q.')
        if action == 'q':
            break
        if action == 's':
            completed += 1
            continue
        if action == 'r':
            reason = ask('Rejection reason', 'Unusable passage or unsupported labels')
            record = {'index': index, 'status': 'REJECTED', 'reason': reason}
            save_decision(args.decisions, record)
            decisions[index] = record
            completed += 1
            continue

        candidate = dict(draft)
        candidate['characters'] = normalize_names(ask('Characters (comma-separated)', ', '.join(normalize_names(draft.get('characters', [])))))
        for field in ('setting', 'objective', 'premise'):
            candidate[field] = ask(field.capitalize(), str(draft.get(field, '')))
        errors = validate(candidate)
        if errors:
            print('Cannot approve:')
            for error in errors:
                print(' -', error)
            print('This candidate remains unreviewed. Rerun to edit it.')
            completed += 1
            continue
        print('\nVerify objective, premise, setting and characters against the FULL passage.')
        if input('I verified these labels against the passage. Type APPROVE: ').strip() != 'APPROVE':
            print('Not saved.')
            completed += 1
            continue
        candidate['review_status'] = 'APPROVED'
        candidate.pop('review_notes', None)
        candidate.pop('suggested_names_unverified', None)
        record = {'index': index, 'status': 'APPROVED', 'example': candidate}
        save_decision(args.decisions, record)
        decisions[index] = record
        completed += 1
        print('Approved and saved.')

    count = export_approved(decisions, args.output)
    print(f'\nSession finished. Approved total: {count}; rejected total: {sum(d.get("status") == "REJECTED" for d in decisions.values())}')
    print(f'Decisions: {args.decisions}')
    print(f'Approved dataset: {args.output}')
    print('Note: approval is human-verified, not independently machine-certified. Do not train until quality and book-disjoint splits are checked.')


if __name__ == '__main__':
    main()
