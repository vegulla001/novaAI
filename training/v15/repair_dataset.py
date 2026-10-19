"""Build provenance-safe V15 SFT splits and review queue. No AI-generated fake labels."""
import argparse, collections, hashlib, json, random, re
from pathlib import Path

def rows(path):
    p=Path(path)
    if not p.exists(): return []
    out=[]
    for n,line in enumerate(p.read_text(encoding='utf8').splitlines(),1):
        if not line.strip():continue
        try: obj=json.loads(line)
        except Exception as e:raise ValueError(f'{p}:{n}: {e}')
        if not isinstance(obj,dict):raise ValueError(f'{p}:{n}: expected object')
        out.append(obj)
    return out

def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf8') as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')

def norm(s):return re.sub(r'\s+',' ',str(s).lower()).strip()
def fingerprint(s):return hashlib.sha256(norm(s).encode()).hexdigest()
def family(r):return str(r.get('source_family') or r.get('source_book') or r.get('book_id') or r.get('title') or '').strip()

def issues(r):
    errs=[]
    for key in ('characters','setting','obstacle','outcome','story'):
        if not r.get(key):errs.append('missing_'+key)
    if not (r.get('goal') or r.get('objective')):errs.append('missing_goal')
    story=str(r.get('story',''))
    if len(story.split())<60:errs.append('short_story')
    if any(x in story for x in ('<INSTRUCTION>','<STORY>','<GENRE_','<END_STORY>')):errs.append('control_marker_in_story')
    if '\ufffd' in story:errs.append('replacement_character')
    chars=r.get('characters',[])
    if not isinstance(chars,list):errs.append('characters_not_list')
    else:
        for c in chars:
            name=c.get('name','') if isinstance(c,dict) else str(c)
            if name and not re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',story,re.I):errs.append('missing_character_'+name)
    if not family(r):errs.append('missing_source_family')
    return errs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pilot',default='dataset/v12/pilot_expanded')
    ap.add_argument('--verified',default='dataset/v12/curated_verified.jsonl')
    ap.add_argument('--candidates',default='dataset/v12/review_cleaned.jsonl')
    ap.add_argument('--out',default='dataset/v15_curated')
    ap.add_argument('--min-reviewed-train',type=int,default=1000)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    pilot={split:rows(Path(a.pilot)/f'{split}.jsonl') for split in ('train','val','test')}
    approved=[]
    for r in rows(a.verified):
        status=str(r.get('review_status','')).upper()
        # This file has historical 'verified' naming, but only explicitly approved rows may enter SFT.
        if status in ('HUMAN_APPROVED','APPROVED','HUMAN_VERIFIED') or r.get('human_approved') is True:
            approved.append(r)
    # Keep original held-out pilot families intact. Never split examples from a family across splits.
    used_families={split:{family(r) for r in data} for split,data in pilot.items()}
    blocked=used_families['val']|used_families['test']
    for r in approved:
        if family(r) not in blocked:pilot['train'].append(r)
    seen={};accepted={};rejected=[]
    for split in ('val','test','train'):
        accepted[split]=[]
        for r in pilot[split]:
            errs=issues(r);fp=fingerprint(r.get('story',''))
            if fp in seen:errs.append('duplicate_story_in_'+seen[fp])
            if not errs:
                seen[fp]=split;accepted[split].append(r)
            else:rejected.append({'split':split,'source_family':family(r),'issues':errs,'record':r})
    # Ensure no family overlap across splits, even when content differs.
    fam={s:{family(r) for r in rs} for s,rs in accepted.items()}
    overlaps={f'{a}_{b}':sorted(fam[a]&fam[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if any(overlaps.values()):
        raise RuntimeError(f'Family leakage found: {overlaps}. Fix split assignments before training.')
    for s,rs in accepted.items():write(out/f'{s}.jsonl',rs)
    write(out/'rejected.jsonl',rejected)
    queue=[];known=set(seen)
    for r in rows(a.candidates):
        fp=fingerprint(r.get('story',''))
        if fp in known:continue
        known.add(fp)
        queue.append({'review_status':'UNREVIEWED','issues':issues(r),'candidate':r})
    write(out/'review_queue.jsonl',queue)
    report={'reviewed_usable':{s:len(v) for s,v in accepted.items()},'rejected_count':len(rejected),
      'unverified_review_queue':len(queue),'source_family_overlap':overlaps,
      'min_reviewed_train':a.min_reviewed_train,'ready_for_sft':len(accepted['train'])>=a.min_reviewed_train and len(accepted['val'])>=10,
      'warning':'Unverified review_queue entries are excluded from all training splits. Quality checks are heuristics, not human review.'}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
