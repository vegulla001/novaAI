"""Build an editable story bible from user-supplied details; no false claim of AI-written biographies."""
import argparse,json,re
from pathlib import Path
from auto_genres import predict
from genres import normalize_genres

def make_foundation(title,premise,genres,characters,chapters,intro):
    if genres:
        labels=normalize_genres(genres)
        method='user'
    else:
        labels,_=predict(title+' '+premise)
        method='keyword_heuristic'
    people=[]
    for i,name in enumerate(characters or ['Alex','Morgan','Raven']):
        people.append({'name':name,'role':('protagonist' if i==0 else 'ally' if i==1 else 'antagonist'),
                       'age':None,'personality':[],'goal':'','backstory':'','relationships':{},
                       'first_appearance':1 if i==0 else min(i+1,chapters)})
    beats=['Introduce the protagonist and their ordinary world',
           'Introduce the central problem and supporting characters',
           'The protagonist takes a risky first step',
           'Complications deepen and relationships change',
           'A major revelation changes the plan',
           'The conflict reaches its climax',
           'Resolve the consequences and close the story']
    plans=[]
    for i in range(1,chapters+1):
        idx=round((i-1)*(len(beats)-1)/max(chapters-1,1))
        plans.append({'number':i,'title':f'Chapter {i}', 'objective':beats[idx],
                      'characters':[x['name'] for x in people if x['first_appearance']<=i],
                      'location':'','events':[],'outcome':''})
    return {'title':title,'premise':premise,'genres':labels,'genre_source':method,
            'introduction_style':intro,'world':{'setting':'','locations':[],'rules':[]},
            'characters':people,'chapters':plans,'continuity':{'completed_events':[], 'unresolved_threads':[]}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--title',required=True);p.add_argument('--premise',required=True)
    p.add_argument('--genres',nargs='*');p.add_argument('--characters',nargs='*')
    p.add_argument('--chapters',type=int,default=8)
    p.add_argument('--introduction-style',choices=['profiles','natural','prologue'],default='natural')
    p.add_argument('--output',default='output/foundation.json')
    a=p.parse_args()
    if a.chapters<1:p.error('chapters must be positive')
    data=make_foundation(a.title,a.premise,a.genres,a.characters,a.chapters,a.introduction_style)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved',path,'genres:',data['genres'],'source:',data['genre_source'])
    print('Edit character details and chapter events before generation.')
if __name__=='__main__':main()
