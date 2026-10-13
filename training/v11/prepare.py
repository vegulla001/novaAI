import argparse, hashlib, json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizer import BPETokenizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokenizer',default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output',default='dataset/v11/processed')
    p.add_argument('--max-train-stories',type=int,default=100000)
    p.add_argument('--max-val-stories',type=int,default=2000)
    args=p.parse_args()
    tok=BPETokenizer.load(args.tokenizer)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    meta={'tokenizer_sha256':digest,'dataset':'roneneldan/TinyStories','format':'flat uint32 token stream'}
    for split,limit in [('train',args.max_train_stories),('validation',args.max_val_stories)]:
        path=out/('train.bin' if split=='train' else 'val.bin')
        count=0;tokens=0
        with path.open('wb') as f:
            ds=load_dataset('roneneldan/TinyStories',split=split,streaming=True)
            for row in ds:
                story=row['text'].strip()
                if not story:continue
                ids=tok.encode('<STORY>\n'+story+'\n<END_STORY>\n')
                if ids:
                    np.asarray(ids,dtype=np.uint32).tofile(f)
                    tokens+=len(ids);count+=1
                if count>=limit:break
                if count%10000==0:print(split,count,'stories',tokens,'tokens',flush=True)
        meta[split]={'stories':count,'tokens':tokens,'path':str(path)}
        print(split,count,tokens,flush=True)
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
