"""Convert real character-card records into instruction/target JSONL.
Only uses source fields; skips records without narrative target text.
"""
import argparse, json, random
from pathlib import Path

def first_str(row,*names):
    for name in names:
        v=row.get(name)
        if isinstance(v,str) and v.strip(): return v.strip()
    return ''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='dataset/structured/raw.jsonl')
    p.add_argument('--output-dir',default='dataset/structured')
    p.add_argument('--val-ratio',type=float,default=.1)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if not 0<a.val_ratio<1: p.error('val-ratio must be between 0 and 1')
    records=[]; skipped=0
    for line in Path(a.input).open(encoding='utf-8'):
        row=json.loads(line)
        name=first_str(row,'character_name','name')
        character=first_str(row,'character','description','character_description')
        setting=first_str(row,'setting','world')
        outline=first_str(row,'story_outline','outline')
        target=first_str(row,'story_introduction','introduction','story','opening')
        if not target or not (name or character or setting or outline):
            skipped+=1; continue
        instruction='\n'.join(x for x in [
            'Write a story introduction consistent with these details.',
            f'Character name: {name}' if name else '',
            f'Character profile: {character}' if character else '',
            f'Setting: {setting}' if setting else '',
            f'Outline: {outline}' if outline else ''] if x)
        records.append({'instruction':instruction,'response':target,'source':'character_card'})
    if len(records)<2: raise SystemExit('Not enough usable records. Inspect raw fields; dataset schema may differ.')
    random.Random(a.seed).shuffle(records)
    nval=max(1,round(len(records)*a.val_ratio))
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('train',records[nval:]),('val',records[:nval])]:
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(name,len(rows),'->',out/f'{name}.jsonl')
    print('Skipped records:',skipped)
    print('NOTE: This random split is for a pipeline smoke test. Deduplicate and group by source story before reporting validation quality.')
if __name__=='__main__': main()
