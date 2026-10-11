"""Transparent heuristic genre suggestions. No pretrained model; not reliable classification."""
import argparse, json, re
from pathlib import Path
from genres import GENRES

KEYWORDS = {
 'ACTION': ('battle','sword','fight','soldier','attack','warrior'),
 'ADVENTURE': ('journey','voyage','island','quest','travel','expedition'),
 'FANTASY': ('dragon','wizard','magic','kingdom','spell','enchanted'),
 'ROMANCE': ('love','marriage','beloved','heart','kiss','courtship'),
 'MYSTERY': ('mystery','clue','detective','investigation','secret','suspect'),
 'THRILLER': ('pursuit','escape','danger','conspiracy','threat','chase'),
 'HORROR': ('terror','ghost','vampire','haunted','blood','nightmare'),
 'SCI_FI': ('spaceship','planet','machine','scientist','future','experiment'),
 'COMEDY': ('laughter','laugh','joke','absurd','amusing','ridiculous'),
 'DRAMA': ('conflict','betrayal','family','sorrow','grief','suffering'),
 'SLICE_OF_LIFE': ('breakfast','school','neighbor','household','daily','routine'),
 'SUPERNATURAL': ('spirit','ghost','curse','demon','apparition','witch'),
 'HISTORICAL': ('empire','duke','queen','century','monarch','colonel'),
 'TRAGEDY': ('death','fatal','doom','grief','mourning','tragedy'),
}

def predict(text, max_genres=3):
    text = text.lower()
    scores = {g: sum(len(re.findall(r"\b"+re.escape(w)+r"\b",text)) for w in words)
              for g,words in KEYWORDS.items()}
    ranking = sorted(scores, key=lambda g: (-scores[g],g))
    selected = [g for g in ranking if scores[g] > 0][:max_genres]
    return selected or ['DRAMA'], {g: scores[g] for g in ranking[:max_genres]}

def predict_file(path):
    text=Path(path).read_text(encoding='utf-8-sig',errors='replace')
    return predict(text)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',default='dataset/novels')
    p.add_argument('--output',default='dataset/genres_auto.json')
    p.add_argument('--max-genres',type=int,default=3)
    args=p.parse_args()
    results={}
    for f in sorted(Path(args.input_dir).rglob('*.txt')):
        genres,scores=predict(f.read_text(encoding='utf-8-sig',errors='replace'),args.max_genres)
        results[f.name]=genres
        print(f'{f.name}: {genres} (keyword counts: {scores})')
    if not results: raise SystemExit('No .txt files found')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Saved',target,'— heuristic suggestions; review before training.')
if __name__=='__main__': main()
