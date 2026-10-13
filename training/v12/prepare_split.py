"""Prepare an existing family-held-out train/val split without reshuffling or test leakage."""
import argparse, hashlib, json
from pathlib import Path
from tokenizer import BPETokenizer
from training.v12.prepare import format_prompt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',default='dataset/v12/splits')
    p.add_argument('--output',default='dataset/v12/processed')
    p.add_argument('--tokenizer',default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--max-length',type=int,default=512)
    args=p.parse_args()
    tok=BPETokenizer.load(args.tokenizer)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stats={}
    for split in ('train','val'):
        source=Path(args.input_dir)/f'{split}.jsonl'
        rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
        kept=0; skipped=0
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:
                prefix=tok.encode(format_prompt(row)); continuation=tok.encode(row['story'].strip()+'\n<END_STORY>\n')
                ids=prefix+continuation
                if len(ids)>args.max_length:
                    skipped+=1;continue
                if not continuation:continue
                labels=[-1]*(len(prefix)-1)+continuation
                assert len(labels)==len(ids)-1
                f.write(json.dumps({'ids':ids,'labels':labels})+'\n');kept+=1
        if not kept:raise ValueError(f'{split}: no fitting examples; increase --max-length up to model context size')
        stats[split]={'kept':kept,'skipped':skipped}
        print(f'{split}: kept={kept} skipped={skipped}')
    meta={'demo_only':True,'num_examples':sum(v['kept'] for v in stats.values()),'tokenizer_sha256':hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),'max_length':args.max_length,'source':str(args.input_dir),'splits':stats,'reason':'Synthetic template bootstrap, not cleared for GPU training'}
    (out/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Safety gate: demo_only=true. GPU training remains blocked.')
if __name__=='__main__':main()
