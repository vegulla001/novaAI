#!/usr/bin/env python3
"""Rebuild v16 character supervision. Strict heuristic real labels + varied synthetic task.

Output is compatible with training.v16.train_character (response-masked SFT lists).
Not a human semantic verification system. Never edits the original data.
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path
import torch
from tokenizers import Tokenizer

HEADER = 'Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\n'
# Common false-positive 'names' from capitalization. Conservative, not exhaustive.
STOP = set(('Each Every Either Neither Someone Somebody Anyone Everybody Everyone Nobody Nothing Something '
 'And The This That These Those Then There Here After Before Since While When Where What Which Who Why '
 'One Two Three Four Five First Second Third Next Last Many Most More Some All Any Other Another '
 'But Yet For Nor So Or As If I You We He She They Them His Her Our Your My Its It Me Us '
 'No Yes Oh Ah Well Now Today Tomorrow Yesterday Monday Tuesday Wednesday Thursday Friday '
 'Saturday Sunday January February March April May June July August September October November December '
 'Mr Mrs Ms Miss Dr Prof Sir Madam Lady Lord King Queen Prince Princess God Chapter Part Book Scene '
 'Continued Continuation Story Notes Preface Introduction End Once Again Indeed However Suddenly '
 'Dear Good Great Little Young Old New Then Soon Finally Therefore Perhaps Please').split())
NAME = re.compile(r"\b[A-Z][a-z]{2,}(?:['-][A-Z][a-z]+)?\b")
SENT_END = re.compile(r'[.!?][\"\u201d\u2019\']?(?=\s|$)')

def count_name(t, n):
    return len(re.findall(r'(?<!\w)' + re.escape(n) + r'(?!\w)', t))

def real_names(ctx, answer):
    # Name must appear *not only at start of sentences* in the context and
    # at least once in the completion; avoid a capitalized first word.
    eligible=[]
    for name in sorted(set(NAME.findall(ctx)) & set(NAME.findall(answer))):
        if name in STOP or len(name)<3: continue
        occurrences=list(re.finditer(r'(?<!\w)'+re.escape(name)+r'(?!\w)',ctx))
        inside=sum(1 for m in occurrences if ctx[:m.start()].rstrip()[-1:] not in ('','.', '!', '?', '\n', ':'))
        if inside < 1: continue
        if len(occurrences) < 2 and count_name(answer,name)<2: continue
        eligible.append((name, len(occurrences)+count_name(answer,name)))
    return [n for n,_ in sorted(eligible,key=lambda x:(-x[1],x[0]))[:2]]

def encode(tok,prompt,answer,max_tokens):
    enc=tok.encode(prompt+answer,add_special_tokens=False)
    if len(enc.ids)>max_tokens or len(enc.ids)<30: return None
    boundary=len(prompt)
    labs=[-100 if end<=boundary else ident for ident, (_,end) in zip(enc.ids,enc.offsets)]
    if sum(x!=-100 for x in labs)<20 or sum(x==-100 for x in labs)<20: return None
    return {'input_ids':enc.ids,'labels':labs}

def extract_original(tok,row):
    ids=list(row['input_ids']); labels=list(row['labels'])
    if len(ids)!=len(labels):return None
    ix=next((i for i,x in enumerate(labels) if x!=-100),None)
    if ix is None:return None
    before=tok.decode(ids[:ix],skip_special_tokens=True)
    if 'Story so far:' not in before or 'Continuation:' not in before:return None
    context=before.split('Story so far:',1)[1].rsplit('Continuation:',1)[0].strip()
    completion=tok.decode(ids[ix:],skip_special_tokens=True).strip()
    return context,completion

def make_real(tok,rows,max_tokens,cap,seed):
    rng=random.Random(seed); indexes=list(range(len(rows)));rng.shuffle(indexes)
    accepted=[]; stats=Counter(); seen=set()
    for j in indexes:
        parsed=extract_original(tok,rows[j])
        if not parsed:stats['invalid_record']+=1;continue
        context,answer=parsed
        if len(context)<130 or len(answer)<90:stats['too_short']+=1;continue
        # Preserve a complete prompt boundary: truncate at previous sentence ending.
        ends=list(SENT_END.finditer(context))
        if not ends:stats['context_no_sentence_end']+=1;continue
        context=context[:ends[-1].end()].strip()
        # Remove the leading broken sentence from token-window continuations.
        # This sacrifices data but avoids showing "hate him" after "Continuation:".
        ends=list(SENT_END.finditer(answer))
        if len(ends)<2:stats['completion_too_few_sentences']+=1;continue
        answer=answer[ends[0].end():ends[-1].end()].strip()
        if len(answer)<80:stats['completion_too_short']+=1;continue
        names=real_names(context,answer)
        if len(names)!=2:stats['not_two_supported_names']+=1;continue
        key=(context[-180:],answer[:180])
        if key in seen:stats['duplicate']+=1;continue
        seen.add(key)
        prompt=(HEADER+'Characters to preserve: '+', '.join(names)+'\n\nStory so far:\n'+context+'\n\nContinuation:\n')
        row=encode(tok,prompt,answer,max_tokens)
        if not row:stats['token_limit']+=1;continue
        accepted.append(row)
        if len(accepted)>=cap:break
    stats['accepted']=len(accepted)
    return accepted,dict(stats)

TRAIN_NAMES=['Arin','Elara','Mira','Jonas','Nila','Dev','Tarin','Sela','Borin','Lysa','Kael','Rina','Orin','Tessa','Lorin','Fara','Darin','Mara','Varek','Iona','Joren','Lina','Ravel','Yara','Kiran','Sorin','Brina','Evan','Cora','Milan','Talia','Rohan','Jessa','Caden','Seren','Riven','Ayla','Halen','Vera','Kora','Davin']
VAL_NAMES=['Aven','Ilyra','Rovan','Eska','Tovin','Naera','Belen','Osric','Ysola','Damar','Thessa','Noran']
SETTINGS=[('abandoned castle','northern gate','a letter from a missing father'),('stormy harbor','empty pier','a map of the missing boat'),('silent library','locked tunnel','a faded journal'),('snow-covered village','old bridge','a broken compass'),('forest campsite','eastern trail','a small lantern'),('mountain watchtower','broken staircase','a leather journal'),('desert outpost','western archway','a silver key'),('market square','stone fountain','a folded note')]
OPENERS=[
 '{a} and {b} had reached the {place} before dawn. {a} carried {item}, while {b} watched the {landmark}. They had agreed to stay away from the {landmark} until they knew what waited there.',
 'In the {place}, {b} warned {a} about the {landmark}. {a} hid {item} under a coat, unwilling to take a chance. Both had promised to return safely.',
 '{b} followed {a} into the {place}. The {landmark} stood ahead, and {a} held {item}. They heard an unfamiliar sound and stopped without moving closer.',
 '{a} stood with {b} near the {place}. {b} kept an eye on the {landmark}. Earlier that night, {a} had found {item}, but neither knew why it mattered.',
]
ANSWERS=[
 '{a} turned to {b}. "We should not move yet," {a} said. {b} lowered the lantern and watched the {landmark}. The two stayed where they were.',
 '{b} caught {a} by the sleeve. "I heard something," {b} whispered. {a} glanced toward the {landmark} and nodded. Neither was ready to step closer.',
 '"Stay beside me," {a} told {b}. {b} agreed, and the two moved away from the {landmark}. They would search for answers when the danger had passed.',
 '{a} lifted {item} and looked at {b}. "We need to think," {a} said. {b} listened carefully as footsteps echoed near the {landmark}.',
 '{b} stepped back and called to {a}. {a} stopped immediately, remembering their agreement. The two decided to wait before returning to the {landmark}.',
 '"Not that way," {b} warned {a}. {a} looked back at the {landmark} and put {item} away. Together they searched for a safer route.',
]

def synthetic(tok,n,names,max_tokens,seed):
    rng=random.Random(seed); output=[]; used=set(); attempts=0
    while len(output)<n and attempts<max(1000, n*30):
        attempts+=1
        a,b=rng.sample(names,2); place,landmark,item=rng.choice(SETTINGS)
        op=rng.randrange(len(OPENERS)); ans=rng.randrange(len(ANSWERS))
        key=(a,b,place,op,ans)
        if key in used:continue
        used.add(key)
        vars=dict(a=a,b=b,place=place,landmark=landmark,item=item)
        before=OPENERS[op].format(**vars); after=ANSWERS[ans].format(**vars)
        prompt=HEADER+f'Characters to preserve: {a}, {b}\n\nStory so far:\n{before}\n\nContinuation:\n'
        row=encode(tok,prompt,after,max_tokens)
        if row:output.append(row)
    if len(output)!=n:raise RuntimeError(f'Generated {len(output)} of {n} requested controlled records')
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',default='dataset/v15_prose_expanded/continuation_sft')
    p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--out-dir',default='dataset/v16_character_strict_sft')
    p.add_argument('--controlled-train',type=int,default=6000)
    p.add_argument('--controlled-val',type=int,default=400)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args();tok=Tokenizer.from_file(a.tokenizer)
    root=Path(a.source_dir)
    train=torch.load(root/'train_sft.pt',map_location='cpu',weights_only=True)
    val=torch.load(root/'val_sft.pt',map_location='cpu',weights_only=True)
    # Strict 50:50 mixing by construction. No replacement of current v16 files.
    real_train,train_stats=make_real(tok,train,a.max_tokens,a.controlled_train,a.seed)
    real_val,val_stats=make_real(tok,val,a.max_tokens,a.controlled_val,a.seed+1)
    if len(real_train)<500 or len(real_val)<50:
        raise RuntimeError('Strict real-fiction filter yielded too few records. See acceptance rules; do not silently weaken validation.')
    controlled_train=synthetic(tok,len(real_train),TRAIN_NAMES,a.max_tokens,a.seed+2)
    controlled_val=synthetic(tok,len(real_val),VAL_NAMES,a.max_tokens,a.seed+3)
    # Separate sources in split files to support explicit per-type evaluations.
    rng=random.Random(a.seed)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    whole_train=real_train+controlled_train;whole_val=real_val+controlled_val
    rng.shuffle(whole_train);rng.shuffle(whole_val)
    for name,rows in [('train_sft.pt',whole_train),('val_sft.pt',whole_val),('val_real.pt',real_val),('val_controlled.pt',controlled_val)]:
        torch.save(rows,out/name)
    report=dict(train=len(whole_train),val=len(whole_val),train_real=len(real_train),train_controlled=len(controlled_train),val_real=len(real_val),val_controlled=len(controlled_val),train_stats=train_stats,val_stats=val_stats,notes=['Heuristic person-name selection is not human semantic verification','Synthetic templates may be memorized; compare novel writing and held-out names','Train and validation fiction originate from prior splits; no new book-ID split is asserted'])
    (out/'build_report_strict.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('STRICT_CHARACTER_SFT_BUILT',json.dumps(report,indent=2))

if __name__=='__main__': main()
