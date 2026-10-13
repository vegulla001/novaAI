"""Inspect lexical/template repetition; not a substitute for human literary review."""
import argparse,json,re,statistics
from collections import Counter
from pathlib import Path

def words(s):return re.findall(r"\b\w+\b",s.lower())
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='dataset/v12/raw/curated.jsonl');args=p.parse_args()
 rows=[json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines() if s.strip()]
 if not rows:raise ValueError('Empty dataset')
 lengths=[len(words(r['story'])) for r in rows]
 scenes=Counter(str(r.get('scene_family','unknown')) for r in rows)
 ngrams=Counter()
 for r in rows:
  ws=words(r['story'])
  ngrams.update(set(tuple(ws[i:i+5]) for i in range(len(ws)-4)))
 print('Examples:',len(rows),'Independent scene families:',len(scenes))
 print('Story words min/median/max:',min(lengths),statistics.median(lengths),max(lengths))
 print('Most repeated five-word phrases (document frequency):')
 for gram,count in ngrams.most_common(12):print(f'  {count}/{len(rows)}: {" ".join(gram)}')
 print('WARNING: structural checks cannot establish narrative quality, originality of patterns, or instruction-following.')
if __name__=='__main__':main()
