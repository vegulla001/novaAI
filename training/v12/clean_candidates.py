"""Clean literary candidates without treating capitalized words as characters.

Character matches require an explicitly curated per-book allowlist. Output is
review-only: matching a name does not establish its role in the scene.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

PLACE = re.compile(r'\b(?:in|into|through|under|inside|down|near|across) (?:the|a|an) ([\w-]+(?: [\w-]+){0,3} (?:forest|castle|room|garden|river|tunnel|well|house|island|cave|temple|street))\b', re.I)
VERB = re.compile(r'\b(?:said|went|looked|fell|asked|ran|walked|saw|found|thought|opened|followed|heard|entered|took|turned)\b', re.I)

def load_characters(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Character allowlist must be a JSON object mapping book IDs to name lists')
    result = {}
    for book_id, names in data.items():
        if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names):
            raise ValueError(f'Invalid character list for book {book_id}')
        result[str(book_id)] = list(dict.fromkeys(n.strip() for n in names))
    return result

def match_characters(story, known):
    # Longest first, so 'White Rabbit' suppresses overlapping 'Rabbit'.
    found = []
    occupied = []
    for name in sorted(known, key=len, reverse=True):
        for match in re.finditer(r'(?<!\w)' + re.escape(name) + r'(?!\w)', story, re.I):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            found.append((match.start(), name))
            occupied.append(match.span())
            break
    return [name for _, name in sorted(found)][:5]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='dataset/v12/review_candidates.jsonl')
    p.add_argument('--output', default='dataset/v12/review_cleaned.jsonl')
    p.add_argument('--rejected', default='dataset/v12/rejected_candidates.jsonl')
    p.add_argument('--characters-file', default='dataset/v12/known_characters.json')
    p.add_argument('--min-words', type=int, default=90)
    a = p.parse_args()
    if Path(a.input).resolve() in {Path(a.output).resolve(), Path(a.rejected).resolve()}:
        p.error('Input and output paths must differ')
    known_by_book = load_characters(a.characters_file)
    for name in (a.output, a.rejected):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with open(a.input, encoding='utf-8') as src, open(a.output, 'w', encoding='utf-8') as dst, open(a.rejected, 'w', encoding='utf-8') as bad:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            stats['total'] += 1
            row = json.loads(line)
            raw = row.get('story', '')
            story = re.sub(r'\[(?:Illustration|Image)[^\]]*\]', ' ', raw, flags=re.I)
            story = re.sub(r'\s+', ' ', story).strip()
            reasons = []
            if len(story.split()) < a.min_words:
                reasons.append('too_short')
            if len(re.findall(r'\bCHAPTER\s+(?:[IVXLCDM]+|\d+)\b', raw[:2500], re.I)) > 2:
                reasons.append('contents_page')
            if re.search(r'\b(?:THE MILLENNIUM FULCRUM EDITION|Contents CHAPTER|Project Gutenberg)\b', raw[:600], re.I):
                reasons.append('front_matter')
            if not VERB.search(story):
                reasons.append('not_narrative')
            if reasons:
                bad.write(json.dumps({'line': line_no, 'source_book_id': row.get('source_book_id'), 'reasons': reasons}) + '\n')
                stats['rejected'] += 1
                continue
            book_id = str(row.get('source_book_id', ''))
            names = match_characters(story, known_by_book.get(book_id, []))
            location = PLACE.search(story)
            output = {
                'title': row.get('title', ''), 'genres': row.get('genres', []),
                'characters': names, 'setting': location.group(0) if location else '',
                'objective': '', 'premise': '', 'story': story,
                'source_book_id': row.get('source_book_id'), 'source_url': row.get('source_url'),
                'review_status': 'NEEDS_HUMAN_REVIEW',
                'review_notes': 'Character names come only from a per-book allowlist, but are not scene-verified. Setting is a regex suggestion. Manually verify and write objective/premise before training.'
            }
            dst.write(json.dumps(output, ensure_ascii=False) + '\n')
            stats['for_review'] += 1
            if not names:
                stats['no_allowlisted_character_found'] += 1
    print(dict(stats))
    print('Review-only output: DO NOT TRAIN on review_cleaned.jsonl.')

if __name__ == '__main__':
    main()
