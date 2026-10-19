#!/usr/bin/env python3
"""Build a new V15 byte-level BPE tokenizer and token streams from prepared prose."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

SPECIAL = ['<pad>', '<bos>', '<eos>', '<unk>', '<story>', '<chapter>', '<scene>', '<end_scene>']

def records(path):
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            txt = obj.get('text')
            if not isinstance(txt, str) or not txt.strip():
                raise ValueError(f'{path}:{line_no}: missing nonempty text')
            if 'book_id' not in obj:
                raise ValueError(f'{path}:{line_no}: missing book_id')
            yield obj

def texts(path):
    for r in records(path):
        yield r['text']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',default='dataset/v15/prose/train.jsonl')
    p.add_argument('--val',default='dataset/v15/prose/val.jsonl')
    p.add_argument('--outdir',default='dataset/v15/tokenizer')
    p.add_argument('--vocab-size',type=int,default=8192)
    args=p.parse_args()
    train,val=Path(args.train),Path(args.val)
    if not train.exists() or not val.exists():
        p.error('Missing prepared train/val JSONL. Extract the prepared prose package first.')
    train_ids={str(r['book_id']) for r in records(train)}
    val_ids={str(r['book_id']) for r in records(val)}
    if train_ids & val_ids:
        raise RuntimeError('Train/validation book_id overlap: '+str(sorted(train_ids & val_ids)))
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.normalizer=normalizers.NFC()
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=args.vocab_size,min_frequency=2,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=True)
    tokenizer.train_from_iterator(texts(train),trainer=trainer)
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise RuntimeError(f'Actual vocab {tokenizer.get_vocab_size()} != requested {args.vocab_size}')
    tok_path=out/'tokenizer.json';tokenizer.save(str(tok_path))
    bos,eos=tokenizer.token_to_id('<bos>'),tokenizer.token_to_id('<eos>')
    report={'vocab_size':tokenizer.get_vocab_size(),'tokenizer_sha256':sha(tok_path),'train_book_count':len(train_ids),'val_book_count':len(val_ids),'book_overlap':0,'special_tokens':{s:tokenizer.token_to_id(s) for s in SPECIAL},'splits':{}}
    for split,path in [('train',train),('val',val)]:
        arr=[]; count=0
        for r in records(path):
            arr.append(bos)
            arr.extend(tokenizer.encode(r['text'],add_special_tokens=False).ids)
            arr.append(eos)
            count+=1
        if not arr or max(arr)>=args.vocab_size:
            raise RuntimeError(f'Invalid token stream for {split}')
        target=out/f'{split}.npy'
        np.save(target,np.asarray(arr,dtype=np.uint16))
        report['splits'][split]={'records':count,'tokens':len(arr),'file':str(target),'sha256':sha(target)}
        print(f'{split}: {count} records, {len(arr):,} tokens -> {target}')
    sample='Rhea opened the archive door. Pax waited in the corridor.'
    decoded=tokenizer.decode(tokenizer.encode(sample,add_special_tokens=False).ids)
    if decoded != sample:
        raise RuntimeError(f'Round-trip mismatch: {decoded!r}')
    (out/'build_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PASS: book-disjoint splits, tokenizer round-trip, token ID bounds')
    print('Tokenizer SHA256:',report['tokenizer_sha256'])
    print('NOTE: these are pretraining token streams, NOT instruction fine-tuning examples.')

if __name__=='__main__':main()
