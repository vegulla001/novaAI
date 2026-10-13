"""Audit curated StoryAI instruction/story JSONL before spending GPU time."""
import argparse,collections,json,re,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--min-words',type=int,default=30);args=p.parse_args()
 rows=[];errors=[];genres=collections.Counter();seen=set();duplicate=0;missing_char=0
 for line_no,line in enumerate(Path(args.input).read_text(encoding='utf-8').splitlines(),1):
  if not line.strip():continue
  try: r=json.loads(line)
  except json.JSONDecodeError as exc:errors.append(f'line {line_no}: invalid JSON: {exc}');continue
  for key in ('title','characters','setting','objective','story'):
   if not r.get(key):errors.append(f'line {line_no}: missing {key}')
  story=r.get('story','');words=len(story.split())
  if words<args.min_words:errors.append(f'line {line_no}: story too short ({words} words)')
  if '<INSTRUCTION>' in story or '<STORY>' in story:errors.append(f'line {line_no}: story contains control markers')
  chars=r.get('characters',[]);names=[c if isinstance(c,str) else c.get('name','') for c in chars]
  absent=[n for n in names if n and not re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',story,re.I)]
  if absent:missing_char+=1;errors.append(f'line {line_no}: missing character(s) in story: {absent}')
  normalized=re.sub(r'\s+',' ',story.casefold()).strip()
  if normalized in seen:duplicate+=1;errors.append(f'line {line_no}: duplicate story')
  seen.add(normalized)
  for genre in r.get('genres',[r.get('genre','unspecified')]):genres[genre]+=1
  rows.append(r)
 print('Examples:',len(rows),'Genres:',dict(genres),'Duplicate stories:',duplicate,'Missing-character examples:',missing_char)
 if rows:print('Median story words:',statistics.median(len(r.get('story','').split()) for r in rows))
 print('Issues:',len(errors))
 for issue in errors[:40]:print(' -',issue)
 if len(errors)>40:print(' ... and',len(errors)-40,'more')
 print('STATUS:', 'PASS (structural checks only)' if not errors else 'FAIL')
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
