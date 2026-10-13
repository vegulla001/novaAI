"""Export one pilot JSONL example to a JSON file without changing source data."""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='dataset/v12/pilot_expanded/val.jsonl')
    ap.add_argument('--title', default='The Tide Archive')
    ap.add_argument('--output', default='output/v12_scene_plan/tide_record.json')
    a = ap.parse_args()
    matches = [json.loads(line) for line in Path(a.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    found = next((r for r in matches if r.get('title') == a.title), None)
    if found is None:
        raise ValueError(f'Title not found: {a.title!r}')
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved:', target)

if __name__ == '__main__':
    main()
