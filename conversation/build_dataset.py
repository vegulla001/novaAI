import json, random, argparse
from pathlib import Path

NAMES=['Lyra','Arin','Kael','Mira','Elara','Rowan','Sera','Darin','Nora','Vex','Talia','Finn','Zara','Orin']
ROLES=['mage','warrior','healer','villain','ranger','knight','scholar']

def example(rng):
    name=rng.choice(NAMES); role=rng.choice(ROLES)
    op=rng.choice(['ADD_CHARACTER','UPDATE_CHARACTER','SET_PLOT','ADD_CHAPTER','UNDO'])
    if op=='ADD_CHARACTER':
        prompt=rng.choice([f'Add a character named {name} who is a {role}.',f'Introduce {name} as a {role}.',f'Create {name}, a {role}.'])
        action={'action':op,'name':name,'role':role}
    elif op=='UPDATE_CHARACTER':
        prompt=rng.choice([f'Make {name} a {role}.',f'Change {name} into a {role}.',f'Update {name} to be a {role}.'])
        action={'action':op,'name':name,'role':role}
    elif op=='SET_PLOT':
        event=rng.choice(['the kingdom falls','the dragon awakens','the crown is stolen','the heroes escape'])
        prompt=rng.choice([f'Change the plot so {event}.',f'Make the storyline about how {event}.'])
        action={'action':op,'plot':event}
    elif op=='ADD_CHAPTER':
        event=rng.choice(['The Discovery','The Betrayal','The Escape','The Battle'])
        prompt=rng.choice([f'Add a chapter called {event}.',f'Create a new chapter titled {event}.'])
        action={'action':op,'title':event}
    else:
        prompt=rng.choice(['Undo that change.','Revert my last edit.','Undo the last action.'])
        action={'action':op}
    return {'user':prompt,'action':action}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--count',type=int,default=12000); p.add_argument('--out',default='dataset/conversation'); a=p.parse_args()
    rng=random.Random(42); rows=[example(rng) for _ in range(a.count)]; rng.shuffle(rows)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    split=int(len(rows)*.9)
    for name,part in [('train',rows[:split]),('val',rows[split:])]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=True,separators=(',',':'))+'\n' for x in part),encoding='utf-8')
    print(f'Created {split} train and {len(rows)-split} validation samples in {out}')
if __name__=='__main__': main()
