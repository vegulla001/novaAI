#!/usr/bin/env python3
"""Discover fiction candidates from a downloaded Project Gutenberg catalog CSV.
Read-only for existing StoryAI data; produces a review CSV, never auto-approves.
"""
import argparse, csv, json, re
from pathlib import Path
from collections import Counter

GENRES = ('fiction','novel','fantasy','adventure','mystery','detective','science fiction','horror','romance','historical fiction','short stories')
EXCLUDE = ('periodical','magazine','catalog','bibliography','dictionary','encyclopedia','index','poetry','poems','drama','plays','essays','nonfiction','non-fiction')

def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).casefold()).strip()

def find(row, *names):
    cols = {norm(k): v for k, v in row.items() if k}
    for name in names:
        if norm(name) in cols:
            return (cols[norm(name)] or '').strip()
    return ''

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',required=True,help='Downloaded Project Gutenberg CSV metadata')
    p.add_argument('--manifest',default='dataset/v15/sources/manifest.csv')
    p.add_argument('--prose-dir',default='dataset/v15/prose')
    p.add_argument('--out',default='dataset/v15/sources/book_candidates_review.csv')
    p.add_argument('--limit',type=int,default=150)
    a=p.parse_args()
    seen_ids=set(); seen_titles=set()
    for split in ('train','val'):
        path=Path(a.prose_dir)/f'{split}.jsonl'
        if not path.exists(): continue
        for line in path.open(encoding='utf-8'):
            if not line.strip(): continue
            rec=json.loads(line)
            seen_ids.add(str(rec.get('book_id','')).strip())
            if rec.get('title'): seen_titles.add(norm(rec['title']))
    mp=Path(a.manifest)
    if mp.exists():
        with mp.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                seen_ids.add(find(row,'book_id'))
                title=find(row,'title')
                if title: seen_titles.add(norm(title))
    found=[]; stats=Counter()
    with Path(a.catalog).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        print('Catalog columns:',reader.fieldnames)
        for row in reader:
            bid=find(row,'Text#','Text','ebook_id','id','ebook no.')
            title=find(row,'Title')
            authors=find(row,'Authors','Author')
            language=find(row,'Language','Languages')
            subjects=find(row,'Subjects','Subject','Bookshelves','Bookshelf')
            typ=find(row,'Type')
            if not bid or not title: stats['missing_id_or_title']+=1; continue
            if bid in seen_ids or ('pg_'+bid) in seen_ids or norm(title) in seen_titles:
                stats['existing_id_or_title']+=1; continue
            if language and 'en' not in language.lower() and 'english' not in language.lower():
                stats['non_english']+=1; continue
            if typ and norm(typ) not in ('text',''): stats['non_text']+=1; continue
            searchable=norm(title+' '+subjects)
            if any(x in searchable for x in EXCLUDE): stats['excluded_subject']+=1; continue
            if not any(x in searchable for x in GENRES): stats['not_fiction_tagged']+=1; continue
            if len(title)<5: stats['short_title']+=1; continue
            found.append({'book_id':'pg_'+bid,'title':title,'authors':authors,'language':language,
                'subjects':subjects,'source_url':'https://www.gutenberg.org/ebooks/'+bid,
                'file_path':'','license':'UNVERIFIED','approved':'no','split':'train',
                'review_notes':'Verify author/translator/edition rights, work type, duplication, and text quality; use authorized mirror for download'})
            seen_ids.add(bid); seen_titles.add(norm(title))
            if len(found)>=a.limit: break
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    fields=['book_id','title','authors','language','subjects','source_url','file_path','license','approved','split','review_notes']
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(found)
    print(json.dumps({'candidates_for_manual_review':len(found),'output':str(out),'skipped':dict(stats),
        'warning':'Candidate list only. No book was downloaded, licensed, or approved. Title matching is heuristic.'},indent=2))
if __name__=='__main__': main()
