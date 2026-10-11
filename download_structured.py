"""Download a sample of character/story records from a public HF dataset.
Schema is inspected at runtime; no fabricated field values are added.
"""
import argparse, json, itertools
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='G-reen/TheatreLM-v2.1-Characters')
    p.add_argument('--split',default='train')
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--output',default='dataset/structured/raw.jsonl')
    p.add_argument('--trust-remote-code',action='store_true',help='Not supported; inspect dataset scripts manually')
    a=p.parse_args()
    if a.trust_remote_code: p.error('Remote dataset code execution is not supported')
    if a.count<1: p.error('--count must be positive')
    try:
        from datasets import load_dataset
        ds=load_dataset(a.dataset,split=a.split,streaming=True)
    except Exception as e:
        raise SystemExit(f'Could not open dataset: {e}\nCheck dataset availability, split, access requirements and license on Hugging Face.')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with out.open('w',encoding='utf-8') as f:
        try:
            for row in itertools.islice(ds,a.count):
                f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
                if n==0: print('Available fields:',', '.join(row))
                n+=1
        except Exception as e:
            print('Dataset stream stopped:',repr(e))
    print(f'Saved {n} raw records to {out}')
    if n<a.count: print(f'WARNING: requested {a.count}, only received {n}.')
    print('Review license, data quality and output before training.')
if __name__=='__main__': main()
