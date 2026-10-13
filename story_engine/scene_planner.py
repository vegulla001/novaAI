"""Deterministic scene plans and auditable story checks. No model/API required."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

REQUIRED = ('title', 'characters', 'setting', 'objective', 'obstacle', 'outcome')

def plan_scene(record: dict) -> dict:
    missing = [k for k in REQUIRED if not record.get(k)]
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    characters = record['characters']
    if not isinstance(characters, list) or not all(isinstance(x, str) and x.strip() for x in characters):
        raise ValueError('characters must be a nonempty list of names')
    return {
        'title': record['title'],
        'genres': record.get('genres', []),
        'characters': characters,
        'setting': record['setting'],
        'objective': record['objective'],
        'obstacle': record['obstacle'],
        'outcome': record['outcome'],
        'beats': [
            {'id': 'establish', 'requirement': f'Establish {record["setting"]} and introduce {", ".join(characters)}.'},
            {'id': 'goal', 'requirement': f'Show the characters attempting to: {record["objective"]}.'},
            {'id': 'obstacle', 'requirement': f'Make the complication consequential: {record["obstacle"]}.'},
            {'id': 'resolution', 'requirement': f'Resolve the scene with: {record["outcome"]}.'},
        ],
    }

def scene_prompt(plan: dict) -> str:
    # Instruction format is deliberately explicit. This is an experimental prompt,
    # not a claim of byte-for-byte equivalence to generate_novel.py.
    genre = ', '.join(str(g).upper() for g in plan['genres']) or 'FICTION'
    beat_lines = '\n'.join(f'{i}. {b["requirement"]}' for i, b in enumerate(plan['beats'], 1))
    return (f'<INSTRUCTION>\nWrite a {genre} story as narrative prose.\n'
            f'Title: {plan["title"]}\n'
            f'Premise: {plan["objective"]}. {plan["obstacle"]}. {plan["outcome"]}.\n'
            f'Characters: {", ".join(plan["characters"])}\n'
            f'Setting: {plan["setting"]}\nChapter 1:\n'
            f'Follow these scene beats in order:\n{beat_lines}\n'
            'Write only the scene as narrative prose. Do not add unrelated characters or locations.\n<STORY>\n')

def audit_story(text: str, plan: dict) -> dict:
    # Only exact character-name presence is automatic. Semantic adherence needs review.
    mentions = {name: bool(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I))
                for name in plan['characters']}
    words = re.findall(r"\b[\w']+\b", text.lower())
    tri = list(zip(words, words[1:], words[2:]))
    repetition = 0 if not tri else 1 - len(set(tri)) / len(tri)
    return {'character_mentions': mentions,
            'all_characters_mentioned': all(mentions.values()),
            'word_count': len(words),
            'repeated_trigram_fraction': round(repetition, 4),
            'manual_review_required': ['setting', 'objective', 'obstacle', 'outcome', 'event_order', 'coherence'],
            'note': 'Name checks and repetition are heuristics, not proof of story adherence.'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', required=True, help='JSON file containing one story record')
    ap.add_argument('--output-dir', default='output/v12_scene_plan')
    ap.add_argument('--story', help='Optional existing generated .txt to audit')
    args = ap.parse_args()
    r = json.loads(Path(args.record).read_text(encoding='utf-8'))
    plan = plan_scene(r)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scene_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'scene_prompt.txt').write_text(scene_prompt(plan), encoding='utf-8')
    if args.story:
        result = audit_story(Path(args.story).read_text(encoding='utf-8'), plan)
        (out / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print('Saved plan and prompt in:', out)

if __name__ == '__main__':
    main()
