"""Decode V12 prose windows with the ORIGINAL 1024-token tokenizer.
Run in the StoryAI project root. Outputs are for inspection/retokenization,
not reconstructed full books: windows may overlap or be incomplete.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
from tokenizer import BPETokenizer

p=argparse.ArgumentParser()
p.add_argument('--corpus',default='storyai-prose-corpus.zip')
p.add_argument('--tokenizer',default='dataset/processed/tokenizer.json')
p.add_argument('--outdir',default='dataset/v15_decoded_windows')
a=p.parse_args()
with zipfile.ZipFile(a.corpus) as z:
 meta=json.loads(z.read('meta.json'))
 actual=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
 if actual!=meta['tokenizer_sha256']:
  raise SystemExit(f'Tokenizer hash mismatch: actual {actual}, expected {meta["tokenizer_sha256"]}')
 tok=BPETokenizer.load(a.tokenizer)
 out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 for split in ('train','val'):
  count=0
  with z.open(split+'.jsonl') as src, (out/(split+'.jsonl')).open('w',encoding='utf-8') as dst:
   for index,line in enumerate(src):
    row=json.loads(line); ids=row['ids']
    if not ids or any(not isinstance(x,int) or not 0<=x<1024 for x in ids):
     raise ValueError(f'Invalid token IDs in {split} row {index}')
    story=tok.decode(ids)
    dst.write(json.dumps({'book_id':str(row['book_id']),'window_index':index,'text':story},ensure_ascii=False)+'\n')
    count+=1
  print(f'{split}: decoded {count} windows to {out/(split+".jsonl")}')
 print('These are windows, not full reconstructed books. Inspect text before training a new tokenizer.')
