"""Create REVIEW CANDIDATES from book prose; never assert invented labels as verified."""
import argparse,json,re
from pathlib import Path
from collections import Counter

STOP={'The','And','But','For','With','That','This','Then','When','After','Before','There','They','Their','What','Where','From','Into','Upon','Once','Could','Would','Should','Chapter','Mr','Mrs','Miss','I','He','She','We','It','You','His','Her','Our','Your','A','An','In','On','At','Of','To','As','Is','Was','Had','Not'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='dataset/v12/public_domain/prose_segments.jsonl')
    p.add_argument('--output',default='dataset/v12/review_candidates.jsonl')
    p.add_argument('--max-candidates',type=int,default=2000)
    p.add_argument('--min-words',type=int,default=120)
    a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    total=0
    with open(a.input,encoding='utf-8') as f, out.open('w',encoding='utf-8') as w:
        for line in f:
            if total>=a.max_candidates:break
            row=json.loads(line)
            if row.get('type')!='public_domain_prose_not_instruction' or row.get('genre')=='other':continue
            story=row['text'].strip()
            if len(story.split())<a.min_words:continue
            # Names are only suggestions; capitalized words can be sentence starters or places.
            candidates=[n for n,c in Counter(re.findall(r'\b[A-Z][a-z]{2,}\b',story)).most_common() if n not in STOP and c>=2]
            sample={'title':row['title'],'genres':[row['genre']],
                    'characters':[], 'setting':'','objective':'','premise':'',
                    'story':story,'source_book_id':row['book_id'],'source_url':row.get('source_url',''),
                    'suggested_names_unverified':candidates[:10],
                    'review_status':'NEEDS_HUMAN_REVIEW',
                    'review_notes':'Fill characters, setting, objective, premise ONLY with facts supported by the excerpt. Reject weak segments.'}
            w.write(json.dumps(sample,ensure_ascii=False)+'\n');total+=1
    print(f'Wrote {total} UNVERIFIED candidates to {out}; not suitable for instruction training.')
if __name__=='__main__':main()
