#!/usr/bin/env python3
"""Conservative reproducible prose preparation. Python stdlib only."""
import argparse, collections, hashlib, json, pathlib, random, re, unicodedata

def norm(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\[(?:Illustration|Illustrations|Frontispiece|Plate)(?:[^\]]{0,80})\]', ' ', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fingerprint(text):
    return hashlib.sha256(re.sub(r'\W+', ' ', text.casefold()).strip().encode('utf-8')).hexdigest()

def prepare(source, out, seed=42):
    out=pathlib.Path(out); out.mkdir(parents=True,exist_ok=True)
    records=[]; invalid=collections.Counter(); bybook=collections.Counter(); seen=set(); source_books={}
    for line_no,line in enumerate(pathlib.Path(source).open(encoding='utf-8-sig'),1):
        try: rec=json.loads(line)
        except json.JSONDecodeError: invalid['invalid_json']+=1;continue
        if not isinstance(rec,dict) or not isinstance(rec.get('text'),str) or rec.get('book_id') is None:
            invalid['missing_fields']+=1;continue
        if rec.get('type') not in ('public_domain_prose_not_instruction',None):
            invalid['unexpected_type']+=1;continue
        t=norm(rec['text'])
        if len(t)<160: invalid['short']+=1;continue
        fp=fingerprint(t)
        if fp in seen: invalid['exact_normalized_duplicate']+=1;continue
        seen.add(fp)
        bid=str(rec['book_id']); bybook[bid]+=1
        source_books[bid]={'title':rec.get('title'), 'genre':rec.get('genre'), 'source_url':rec.get('source_url')}
        records.append({'text':t,'book_id':rec['book_id'],'title':rec.get('title'),'genre':rec.get('genre'),'source_url':rec.get('source_url'),'type':'public_domain_prose_not_instruction'})
    if len(bybook)<2: raise ValueError('At least two distinct books required for leakage-safe split')
    # Seeded book-level split. Approximate 10% by character count, never split a book.
    totals=collections.Counter()
    for r in records: totals[str(r['book_id'])]+=len(r['text'])
    rng=random.Random(seed); ids=list(totals);rng.shuffle(ids)
    target=sum(totals.values())*.10; selected=[]; running=0
    for bid in ids:
        if len(selected)>=max(1,len(ids)//5) or (selected and running>=target):break
        selected.append(bid);running+=totals[bid]
    valbooks=set(selected)
    if len(valbooks)==len(ids): valbooks.remove(ids[-1])
    splits={'train':[],'val':[]}
    for r in records:splits['val' if str(r['book_id']) in valbooks else 'train'].append(r)
    for split,rows in splits.items():
        with (out/f'{split}.jsonl').open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={'source':str(source),'method':'NFC, illustration marker removal, whitespace normalization, exact normalized deduplication; book-level seeded holdout',
            'seed':seed,'segments_kept':len(records),'excluded':dict(invalid),'books':len(bybook),
            'train':{'segments':len(splits['train']),'books':sorted(set(str(r['book_id']) for r in splits['train'])),'characters':sum(len(r['text']) for r in splits['train'])},
            'val':{'segments':len(splits['val']),'books':sorted(valbooks),'characters':sum(len(r['text']) for r in splits['val'])},
            'source_books':source_books,
            'warnings':['Text may still contain inline tables of contents, front matter, OCR artifacts, or near duplicates.',
                        'Public-domain status and rights in each jurisdiction have not been independently verified.',
                        'No instruction examples were generated or reviewed.',
                        'Do not use validation data to fit the tokenizer; train tokenizer on train.jsonl only.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='dataset/v12/public_domain/prose_segments.jsonl');p.add_argument('--outdir',default='dataset/v15/prose');p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.outdir,a.seed),indent=2,ensure_ascii=False))
