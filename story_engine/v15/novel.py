"""Integrated StoryAI V15: NLU edits, persistent story state, GPT scenes, strict rejection."""
import argparse,json,re,hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import GPT,GPTConfig
from training.v15.train_instruction import prompt as training_prompt
from story_engine.v14.novel import make_outline,names_from

def edit_spec(spec,commands,nlu_checkpoint):
    from nlu.inference import StoryNLU
    nlu=StoryNLU(checkpoint=nlu_checkpoint)
    applied=[];rejected=[]
    for cmd in commands:
        decision=nlu.interpret(cmd,spec)
        action=decision.get('action')
        if action=='ADD_CHARACTER':
            if any(c.get('name','').casefold()==decision['name'].casefold() for c in spec.get('characters',[])):
                rejected.append({'command':cmd,'reason':'Duplicate character'});continue
            spec.setdefault('characters',[]).append({'name':decision['name'],'role':decision['role']})
        elif action=='UPDATE_CHARACTER':
            for c in spec.get('characters',[]):
                if c['name'].casefold()==decision['name'].casefold():c['role']=decision['role']
        elif action=='SET_PLOT':spec['goal']=decision['plot']
        elif action=='ADD_RELATIONSHIP':
            spec.setdefault('relationships',[]).append({k:decision[k] for k in ('character1','character2','relationship')})
        elif action=='ADD_CHAPTER':spec.setdefault('chapter_titles',[]).append(decision['title'])
        else:
            rejected.append({'command':cmd,'decision':decision});continue
        applied.append({'command':cmd,'decision':decision})
    return applied,rejected

def load(checkpoint,tokenizer):
    sha=hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('tokenizer_sha256')!=sha:raise ValueError('Tokenizer SHA256 mismatch')
    tok=Tokenizer.from_file(tokenizer)
    cfg=GPTConfig(**ck['config'])
    if cfg.vocab_size!=tok.get_vocab_size():raise ValueError('Vocabulary mismatch')
    model=GPT(cfg);model.load_state_dict(ck['model']);model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return model,tok

@torch.inference_mode()
def generate(model,tok,text,max_new_tokens,temperature,top_k):
    ids=tok.encode(text).ids
    if len(ids)>=model.config.block_size:raise ValueError('Prompt exceeds model context; refusing to drop grounding')
    output=[];device=next(model.parameters()).device
    for _ in range(min(max_new_tokens,model.config.block_size-len(ids))):
        logits,_=model(torch.tensor([ids+output],device=device))
        z=logits[0,-1].float()/max(temperature,.05)
        if top_k>0:
            cutoff=torch.topk(z,min(top_k,z.numel())).values[-1];z[z<cutoff]=-float('inf')
        nxt=int(torch.multinomial(z.softmax(-1),1));output.append(nxt)
        decoded=tok.decode(output)
        if '<END_STORY>' in decoded:break
    return tok.decode(output).split('<END_STORY>')[0].strip()

def validate(text,names):
    problems=[]
    if len(text.split())<70:problems.append('too_short')
    if re.search(r'<(?:GENRE|INSTRUCTION|STORY|END_STORY)',text):problems.append('leaked_control_markers')
    if text.count('\ufffd')>0:problems.append('invalid_utf8')
    missing=[n for n in names if not re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',text,re.I)]
    if missing:problems.append('missing_characters:'+','.join(missing))
    if len(text.split())>=30:
        words=re.findall(r'[a-z]+',text.lower());unique=len(set(words))/max(1,len(words))
        if unique<.22:problems.append('extreme_repetition')
    return problems

def run(spec,commands,checkpoint,tokenizer,nlu_checkpoint,outdir,max_new_tokens,temperature,top_k):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    applied,rejected=edit_spec(spec,commands,nlu_checkpoint) if commands else ([],[])
    (out/'nlu_report.json').write_text(json.dumps({'applied':applied,'rejected':rejected},indent=2),encoding='utf-8')
    (out/'story_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    model,tok=load(checkpoint,tokenizer);outline=make_outline(spec);names=names_from(spec)
    state={'characters':spec.get('characters',[]),'relationships':spec.get('relationships',[]),'events':[],'chapters':[]}
    (out/'outline.json').write_text(json.dumps(outline,indent=2),encoding='utf-8')
    failures=[];chapter_files=[]
    for chapter in outline:
        scenes=[]
        for scene in range(1,int(spec.get('scenes_per_chapter',2))+1):
            # EXACT same fields/order as instruction SFT. Chapter state is tracked separately; never invent fake continuity.
            row={'characters':spec.get('characters',[]),'genre':spec.get('genre','fiction'),
                 'setting':spec.get('setting',''),'goal':chapter['objective'],
                 'obstacle':chapter['obstacle'],'outcome':chapter['intended_outcome']}
            text=generate(model,tok,training_prompt(row),max_new_tokens,temperature,top_k)
            problems=validate(text,names)
            if problems:
                failures.append({'chapter':chapter['number'],'scene':scene,'issues':problems})
                scenes.append(f'[REJECTED MODEL SCENE: {", ".join(problems)}]')
            else:
                scenes.append(text)
                state['events'].append({'chapter':chapter['number'],'scene':scene,'excerpt':text[:300],
                                        'warning':'Events are generated text, not fact-checked continuity.'})
        title=spec.get('chapter_titles',[])
        label=title[chapter['number']-1] if len(title)>=chapter['number'] else chapter['stage']
        content=f"# Chapter {chapter['number']}: {label}\n\n"+'\n\n'.join(scenes)+'\n'
        file=f"chapter_{chapter['number']:02d}.md";(out/file).write_text(content,encoding='utf-8');chapter_files.append(file)
        state['chapters'].append({'number':chapter['number'],'file':file})
        (out/'story_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    (out/'novel.md').write_text('# '+spec.get('title','Untitled')+'\n\n'+'\n'.join((out/f).read_text() for f in chapter_files),encoding='utf-8')
    report={'quality_gate_passed':not failures and not rejected,'rejected_scenes':failures,'rejected_nlu_commands':rejected,
            'note':'This is a basic quality gate, NOT a measure of prose quality or true chapter continuity.'}
    (out/'quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',default='story_spec.example.json')
    p.add_argument('--commands',help='Text file, one NLU edit command per line')
    p.add_argument('--nlu-checkpoint',default='checkpoints/storyai_v10_nlu.pt')
    p.add_argument('--checkpoint',default='checkpoints/storyai_v15_instruction.pt')
    p.add_argument('--tokenizer',default='dataset/v15/tokenizer.json');p.add_argument('--outdir',default='output/v15_novel')
    p.add_argument('--max-new-tokens',type=int,default=450);p.add_argument('--temperature',type=float,default=.7);p.add_argument('--top-k',type=int,default=30)
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text(encoding='utf-8'))
    commands=Path(a.commands).read_text(encoding='utf-8').splitlines() if a.commands else []
    run(spec,[c for c in commands if c.strip()],a.checkpoint,a.tokenizer,a.nlu_checkpoint,a.outdir,a.max_new_tokens,a.temperature,a.top_k)
if __name__=='__main__':main()
