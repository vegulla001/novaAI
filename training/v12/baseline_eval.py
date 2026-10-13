"""Reproducible, low-cost V8 baseline for StoryAI's existing generate_novel.py.

Run from the repository root:
 python -m training.v12.baseline_eval --checkpoint checkpoints/storyai_v8_best.pt

Only simple literal checks are automated; inspect the stories manually for coherence.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = [
    {"id":"forest_temple", "title":"The Hidden Temple", "premise":"Leo and Arin enter a forbidden forest and discover an ancient temple guarded by a magical barrier.", "genres":["fantasy","adventure"], "characters":["Leo","Arin"], "terms":[["forest","woods","trees"],["temple","shrine","ruins"],["barrier","ward","shield"]]},
    {"id":"ship_storm", "title":"Storm Passage", "premise":"Mara and Tovin steer a damaged ship through a storm and rescue a sailor from the water.", "genres":["adventure"], "characters":["Mara","Tovin"], "terms":[["ship","vessel","boat"],["storm","thunder","waves"],["rescue","saved","pulled"]]},
    {"id":"clockwork", "title":"The Silent Clock", "premise":"Nia investigates a clock tower that stopped at midnight and discovers a hidden mechanism.", "genres":["mystery","fantasy"], "characters":["Nia"], "terms":[["clock","tower"],["midnight","twelve"],["mechanism","gears","machine"]]},
    {"id":"mars_signal", "title":"The Red Signal", "premise":"Iris and Sol decode a distress signal in a Mars research station while oxygen runs low.", "genres":["sci_fi"], "characters":["Iris","Sol"], "terms":[["Mars","red planet"],["signal","transmission","message"],["oxygen","air"]]},
    {"id":"mountain_bridge", "title":"The Broken Crossing", "premise":"Kiran and Dev must cross a broken rope bridge above a mountain gorge before sunset.", "genres":["adventure"], "characters":["Kiran","Dev"], "terms":[["bridge","crossing"],["gorge","ravine","chasm"],["sunset","dusk","dark"]]},
    {"id":"library_secret", "title":"The Missing Page", "premise":"Elena finds a missing page in an old library and realizes her mentor hid a dangerous secret.", "genres":["mystery"], "characters":["Elena"], "terms":[["library","books"],["page","paper"],["mentor","teacher"],["secret","hidden"]]},
]

def contains(text, phrase):
    return bool(re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text, flags=re.IGNORECASE))

def score(text, case):
    chars = {name: contains(text,name) for name in case['characters']}
    terms = {" / ".join(group): any(contains(text,t) for t in group) for group in case['terms']}
    return {"characters":chars, "term_groups":terms,
            "character_recall":sum(chars.values())/len(chars),
            "term_group_recall":sum(terms.values())/len(terms),
            "word_count":len(text.split()),
            "note":"Literal mentions only; does not verify events, coherence, or instruction following."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/storyai_v8_best.pt')
    ap.add_argument('--output-dir',default='output/v12_baseline_v8')
    ap.add_argument('--tokens-per-chapter',type=int,default=350)
    ap.add_argument('--temperature',type=float,default=0.7)
    ap.add_argument('--top-k',type=int,default=20)
    ap.add_argument('--dry-run',action='store_true',help='Print commands without running model')
    a=ap.parse_args()
    if not a.dry_run and not Path(a.checkpoint).is_file():
        ap.error(f'Checkpoint missing: {a.checkpoint}')
    if not Path('generate_novel.py').is_file():
        ap.error('Run this module from the StoryAI repository root (generate_novel.py not found).')
    out=Path(a.output_dir)
    if not a.dry_run: out.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in PROMPTS:
        path=out/(case['id']+'.txt')
        cmd=[sys.executable,'generate_novel.py','--checkpoint',a.checkpoint,
             '--title',case['title'],'--premise',case['premise'],
             '--genres',*case['genres'],'--chapters','1',
             '--tokens-per-chapter',str(a.tokens_per_chapter),
             '--temperature',str(a.temperature),'--top-k',str(a.top_k),
             '--output',str(path)]
        print('\nCASE:',case['id'],flush=True)
        if a.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue
        result=subprocess.run(cmd,capture_output=True,text=True,errors='replace')
        if result.returncode!=0 or not path.is_file():
            records.append({'id':case['id'],'error':result.stderr[-2500:] or result.stdout[-2500:], 'returncode':result.returncode})
            print('FAILED:',case['id'],result.stderr[-500:],flush=True)
            continue
        story=path.read_text(encoding='utf-8',errors='replace')
        rec={'id':case['id'],'prompt':case,'file':str(path),'scores':score(story,case)}
        records.append(rec)
        print('Character recall:',rec['scores']['character_recall'],'Term recall:',rec['scores']['term_group_recall'],flush=True)
    if a.dry_run: return
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':a.checkpoint,
            'settings':{'tokens_per_chapter':a.tokens_per_chapter,'temperature':a.temperature,'top_k':a.top_k},
            'results':records,'manual_review_required':True,
            'warning':'Do not interpret lexical scores as narrative adherence or coherence.'}
    (out/'baseline_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'prompts.json').write_text(json.dumps(PROMPTS,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nSaved:',out/'baseline_report.json')
    print('Read all generated .txt files and score story quality manually before any V12 training.')
if __name__=='__main__':main()
