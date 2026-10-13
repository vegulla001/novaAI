"""Book-disjoint, context-windowed prose LM data. Does NOT make instruction labels."""
import argparse, hashlib, json, random, re
from collections import defaultdict
from pathlib import Path
from tokenizer import BPETokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='dataset/v12/public_domain/prose_segments.jsonl')
    p.add_argument('--tokenizer',default='dataset/processed_structured/tokenizer.json')
    p.add_argument('--output',default='dataset/v12/prose_processed')
    p.add_argument('--block-size',type=int,default=512)
    p.add_argument('--val-books',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-segments',type=int,default=0,help='0 means all')
    a=p.parse_args()
    if a.block_size<64: p.error('block-size must be >=64')
    tok=BPETokenizer.load(a.tokenizer)
    by_book=defaultdict(list)
    with open(a.input,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line)
            if row.get('type')!='public_domain_prose_not_instruction':continue
            if row.get('genre')=='other':continue
            content=re.sub(r'\s+',' ',row.get('text','')).strip()
            if len(content.split())<50:continue
            by_book[str(row['book_id'])].append(content)
            if a.max_segments and sum(map(len,by_book.values()))>=a.max_segments:break
    books=sorted(by_book)
    if len(books)<a.val_books+1:raise ValueError('Not enough distinct books for book-disjoint split')
    random.Random(a.seed).shuffle(books)
    val_books=set(books[:a.val_books]);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    counts={}; lengths={}
    for split,selected in [('train',[b for b in books if b not in val_books]),('val',[b for b in books if b in val_books])]:
        n=0;nt=0
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as fp:
            for book in selected:
                for segment in by_book[book]:
                    ids=tok.encode(segment+'\n')
                    # Independently framed windows, no cross-book or cross-segment leakage.
                    for i in range(0,len(ids)-1,a.block_size):
                        window=ids[i:i+a.block_size+1]
                        if len(window)<33:continue
                        fp.write(json.dumps({'ids':window,'book_id':book})+'\n')
                        n+=1;nt+=len(window)
        counts[split]=n;lengths[split]=nt
    meta={'kind':'prose_lm','tokenizer_sha256':hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest(),
          'block_size':a.block_size,'train_windows':counts['train'],'val_windows':counts['val'],
          'train_tokens':lengths['train'],'val_tokens':lengths['val'],
          'val_books':sorted(val_books),'train_books':sorted(set(books)-val_books),
          'source':a.input}
    (out/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))
    if min(counts.values())==0:raise ValueError('Empty split')
if __name__=='__main__':main()
