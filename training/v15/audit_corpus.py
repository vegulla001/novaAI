"""Audit a text fiction corpus before BPE training. Read-only."""
import argparse,collections,hashlib,json,re
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--corpus',default='dataset/v12/prose_processed');p.add_argument('--out',default='dataset/v15_curated/corpus_audit.json');a=p.parse_args()
 root=Path(a.corpus);files=sorted(root.rglob('*.txt')) if root.exists() else []
 seen={};results=[];dups=[];total=0
 for f in files:
  txt=f.read_text(encoding='utf8',errors='replace');total+=len(txt);digest=hashlib.sha256(re.sub(r'\s+',' ',txt).strip().encode()).hexdigest()
  if digest in seen:dups.append([str(f),seen[digest]])
  else:seen[digest]=str(f)
  results.append({'file':str(f),'characters':len(txt),'approx_words':len(txt.split()),'replacement_chars':txt.count('\ufffd'),'control_markers':len(re.findall(r'<(?:INSTRUCTION|STORY|END_STORY|GENRE_[A-Z_]+)>',txt)),'possible_gutenberg_header':'project gutenberg' in txt[:4000].lower()})
 report={'documents':len(files),'characters':total,'exact_normalized_duplicates':dups,'files':results,'warning':'Does not establish copyright/license permission or detect near-duplicate passages.'}
 dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
if __name__=='__main__':main()
