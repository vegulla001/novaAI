"""Synthetic supervised NLU data. Names/templates are disjoint across train/test."""
import json, random, argparse
from pathlib import Path

INTENTS = ['ADD_CHARACTER','UPDATE_CHARACTER','ADD_RELATIONSHIP','SET_PLOT','ADD_CHAPTER','UNDO','UNKNOWN']
TAGS = ['O','B-CHAR','I-CHAR','B-ROLE','I-ROLE','B-POSITION','I-POSITION','B-RELATION','I-RELATION','B-PLOT','I-PLOT','B-TITLE','I-TITLE','B-REF','I-REF']
TRAIN_NAMES=['Lyra','Arin','Leo','Mira','Kael','Nora','Elara','Finn','Zara','Rowan','Talia','Sera','Darin','Orin','Vex','Iris','Asha','Niko','Luna','Dante']
VAL_NAMES=['Seraphina','Jasper','Amara','Cassian','Thalia','Ezekiel','Raven','Sol','Freya','Briar']
ROLES=['mage','warrior','hero','villain','knight','ranger','healer','wizard','scholar','queen','king','prince','princess']
POSITIONS=['protagonist','antagonist','main character','supporting character']
RELATIONS=['teammates','friends','enemies','siblings','allies']
EVENTS=['the dragon awakens','the kingdom falls','the crown is stolen','the heroes escape','a hidden city is discovered']
TITLES=['The Awakening','The Lost Crown','The First Battle','A Dark Secret','The Final Gate']

def make(text,intent,slots):
    tags=['O']*len(text)
    cursor=0
    for label,value in slots:
        pos=text.find(value,cursor)
        if pos<0: raise ValueError((text,label,value))
        for j in range(pos,pos+len(value)): tags[j]=('B-' if j==pos else 'I-')+label
        cursor=pos+len(value)
    return {'text':text,'intent':intent,'tags':tags}

def sample(rng,names,heldout=False):
    a,b=rng.sample(names,2); role=rng.choice(ROLES); pos=rng.choice(POSITIONS)
    kind=rng.choice(INTENTS)
    if kind=='ADD_CHARACTER':
        if heldout:
            t=f'Please introduce {a} as a {role}.'; slots=[('CHAR',a),('ROLE',role)]
        else:
            pattern=rng.randrange(4)
            if pattern==0: t=f'Add character {a} as {role}';slots=[('CHAR',a),('ROLE',role)]
            elif pattern==1:t=f'Create a character named {a} who is a {role}.';slots=[('CHAR',a),('ROLE',role)]
            elif pattern==2:t=f'Introduce {a} as a {role}';slots=[('CHAR',a),('ROLE',role)]
            else:t=f'Add character {a} as {role} and {pos} of the story';slots=[('CHAR',a),('ROLE',role),('POSITION',pos)]
    elif kind=='UPDATE_CHARACTER':
        if heldout:t=f'Turn {a} into a {role}.';slots=[('CHAR',a),('ROLE',role)]
        elif rng.random()<.25:t=f'Make her a {role}';slots=[('REF','her'),('ROLE',role)]
        else:t=rng.choice([f'Make {a} a {role}',f'Change {a} into a {role}',f'Update {a} to be a {role}']);slots=[('CHAR',a),('ROLE',role)]
    elif kind=='ADD_RELATIONSHIP':
        rel=rng.choice(RELATIONS)
        if heldout:t=f'I want {a} and {b} to become {rel}.'
        elif rel=='teammates' and rng.random()<.5:t=f'Make {a} and {b} as same team'
        else:t=f'Make {a} and {b} {rel}'
        slots=[('CHAR',a),('CHAR',b)]
        if rel in t:slots.append(('RELATION',rel))
    elif kind=='SET_PLOT':
        event=rng.choice(EVENTS);t=(f'The plot should be about {event}' if heldout else f'Change the plot so {event}');slots=[('PLOT',event)]
    elif kind=='ADD_CHAPTER':
        title=rng.choice(TITLES);t=(f'I need a chapter titled {title}' if heldout else f'Add a chapter called {title}');slots=[('TITLE',title)]
    elif kind=='UNDO':t=rng.choice(['Undo that change','Revert my last edit','Undo the last action']);slots=[]
    else:t=rng.choice(['Hello StoryAI','What time is it?','Tell me a joke','I like mangoes']);slots=[]
    return make(t,kind,slots)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-count',type=int,default=14000);p.add_argument('--val-count',type=int,default=1500);p.add_argument('--out',default='dataset/nlu');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for filename,count,names,heldout,seed in [('train',args.train_count,TRAIN_NAMES,False,43),('val',args.val_count,VAL_NAMES,True,44)]:
        rng=random.Random(seed)
        with (out/(filename+'.jsonl')).open('w',encoding='utf-8') as f:
            for _ in range(count):f.write(json.dumps(sample(rng,names,heldout),ensure_ascii=False)+'\n')
    print('Created datasets in',out)
if __name__=='__main__':main()
