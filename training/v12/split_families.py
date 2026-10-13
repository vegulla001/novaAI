"""Split raw JSONL by scene family to prevent near-identical templates crossing splits."""
import argparse, json, random
from collections import Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', default='dataset/v12/raw/curated.jsonl')
    p.add_argument('--output', default='dataset/v12/splits')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-families',type=int,default=3)
    p.add_argument('--test-families',type=int,default=3)
    args=p.parse_args()
    rows=[json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows: raise ValueError('Empty input')
    if any('scene_family' not in row for row in rows):
        raise ValueError('Every row must have scene_family; do not use this split script for unrelated corpora')
    groups=sorted(set(str(r['scene_family']) for r in rows))
    if len(groups) <= args.val_families + args.test_families:
        raise ValueError('Not enough independent scene families')
    random.Random(args.seed).shuffle(groups)
    test=set(groups[:args.test_families]); val=set(groups[args.test_families:args.test_families+args.val_families]); train=set(groups)-test-val
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for name,keys in [('train',train),('val',val),('test',test)]:
        selected=[r for r in rows if str(r['scene_family']) in keys]
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
        print(f'{name}: {len(selected)} examples; families={sorted(keys)}')
    manifest={'source':args.input,'seed':args.seed,'train_families':sorted(train),'val_families':sorted(val),'test_families':sorted(test), 'warning':'All examples are variants of only 16 templates; not an independent literary corpus.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
