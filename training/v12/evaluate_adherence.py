"""Evaluate generated text against explicit requirements. Lexical checks are weak proxies."""
import argparse,json,re
from pathlib import Path

def present(text,term):
    return bool(re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text,re.I))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='JSONL with story and required_characters, required_terms')
    p.add_argument('--output',default='output/v12_adherence_report.json')
    a=p.parse_args();rows=[]
    with open(a.input,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            obj=json.loads(line);story=obj.get('generated_story',obj.get('story',''))
            chars=obj.get('required_characters',[]);terms=obj.get('required_terms',[])
            if not isinstance(chars,list) or not isinstance(terms,list):raise ValueError('Requirements must be lists')
            checks={s:present(story,s) for s in chars+terms if s}
            rows.append({'title':obj.get('title',''),'passed':sum(checks.values()),'total':len(checks),
                         'missing':[k for k,v in checks.items() if not v],
                         'words':len(story.split())})
    if not rows:raise ValueError('No evaluation examples')
    total=sum(x['total'] for x in rows);passed=sum(x['passed'] for x in rows)
    report={'examples':len(rows),'lexical_recall':passed/total if total else None,'checks':total,'details':rows,
            'warning':'Lexical presence does not prove plot adherence, coherence, or continuity. Human review is required.'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Lexical checks: {passed}/{total} ({passed/total:.1%})' if total else 'No requirements provided')
    print('Report:',out)
if __name__=='__main__':main()
