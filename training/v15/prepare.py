"""Train NEW BPE from local fiction, tokenize split by document (not token), and audit data."""
import argparse,hashlib,json,random,re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders

def files_at(root):
    paths=sorted(Path(root).rglob('*.txt'))
    paths=[p for p in paths if p.stat().st_size>=1000]
    if len(paths)<3:raise ValueError('Need >=3 nontrivial .txt documents to create document-disjoint splits')
    return paths

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',default='dataset/v12/prose_processed')
    p.add_argument('--out',default='dataset/v15')
    p.add_argument('--vocab-size',type=int,default=8192)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--min-train-tokens',type=int,default=5000000)
    a=p.parse_args()
    if a.vocab_size<2048:raise ValueError('Vocabulary must be >=2048')
    paths=files_at(a.corpus)
    random.Random(a.seed).shuffle(paths)
    # Split at document level BEFORE training tokenizer to avoid vocabulary leakage from validation text.
    n_val=max(1,round(len(paths)*.1)); val_files=paths[:n_val];train_files=paths[n_val:]
    tokenizer=Tokenizer(models.BPE(unk_token='[UNK]',byte_fallback=False))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=a.vocab_size,min_frequency=2,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['[UNK]'])
    tokenizer.train([str(p) for p in train_files],trainer)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tokenizer.save(str(out/'tokenizer.json'))
    def encode(files):
        chunks=[]; totals=0
        for file in files:
            txt=file.read_text(encoding='utf-8',errors='replace')
            if '\ufffd' in txt:print('WARNING: replacement character in',file)
            ids=tokenizer.encode(txt).ids
            chunks.append(np.asarray(ids,dtype=np.uint32));totals+=len(ids)
        return np.concatenate(chunks),totals
    train,nt=encode(train_files);val,nv=encode(val_files)
    np.save(out/'train.npy',train);np.save(out/'val.npy',val)
    report={'vocab_size':tokenizer.get_vocab_size(),'train_tokens':nt,'val_tokens':nv,
            'train_documents':[str(p) for p in train_files],'val_documents':[str(p) for p in val_files],
            'tokenizer_sha256':hashlib.sha256((out/'tokenizer.json').read_bytes()).hexdigest(),
            'warning':'These are prose-only data, not verified instruction-following pairs.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('documents')},indent=2))
    if nt<a.min_train_tokens:raise RuntimeError(f'Only {nt:,} train tokens. Need >= {a.min_train_tokens:,} before expensive GPU run; acquire more licensed fiction.')
if __name__=='__main__':main()
