"""StoryAI V14: model-driven scene generation with persistent outline, cast and continuity.
Outline and memory are deterministic orchestration, NOT proof of learned capabilities.
"""
import argparse, hashlib, json, re
from pathlib import Path
import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer


def load_model(checkpoint,tokenizer,device):
    tok=BPETokenizer.load(tokenizer)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=GPTConfig(**ck['config']) if isinstance(ck['config'],dict) else ck['config']
    if tok.vocab_size!=config.vocab_size: raise ValueError('Tokenizer vocabulary mismatch')
    digest=hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest()
    if ck.get('tokenizer_sha256') and ck['tokenizer_sha256']!=digest: raise ValueError('Tokenizer checksum mismatch')
    model=GPT(config); model.load_state_dict(ck['model'],strict=True)
    model.to(device).eval()
    return model,tok

@torch.inference_mode()
def generate(model,tok,prompt,max_new_tokens=350,temperature=.85,top_k=40):
    device=next(model.parameters()).device
    ids=tok.encode(prompt)
    if not ids: raise ValueError('Empty prompt')
    out=[]
    for _ in range(max_new_tokens):
        context=(ids+out)[-model.config.block_size:]
        logits,_=model(torch.tensor([context],dtype=torch.long,device=device))
        logits=logits[0,-1].float()/max(temperature,.05)
        k=min(top_k,logits.numel())
        if k>0:
            cutoff=torch.topk(logits,k).values[-1]
            logits[logits<cutoff]=-float('inf')
        nxt=torch.multinomial(torch.softmax(logits,dim=-1),1).item()
        out.append(nxt)
        if '<END_STORY>' in tok.decode(out): break
    return tok.decode(out).split('<END_STORY>')[0].strip()


def make_outline(spec):
    chapters=int(spec.get('chapters',3))
    if not 1<=chapters<=40: raise ValueError('chapters must be 1..40')
    stages=['Introduction','Inciting incident','Rising complications','Reversal','Climax','Resolution']
    result=[]
    for i in range(chapters):
        stage=stages[min(5,round(i*5/max(chapters-1,1)))]
        result.append({'number':i+1,'stage':stage,'objective':spec.get('goal',''),
                       'obstacle':spec.get('obstacle',''),'intended_outcome':spec.get('outcome','')})
    return result


def names_from(spec):
    result=[]
    for item in spec.get('characters',[]):
        result.append(item.get('name','') if isinstance(item,dict) else str(item))
    return [x for x in result if x]


def run(spec,checkpoint,tokenizer,outdir,max_new_tokens,temperature,top_k):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,tok=load_model(checkpoint,tokenizer,device)
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    outline=make_outline(spec)
    names=names_from(spec)
    state={'title':spec.get('title','Untitled'),'characters':spec.get('characters',[]),
           'world':spec.get('setting',''),'plot_goal':spec.get('goal',''),
           'events':[],'unresolved':[],'chapters':[]}
    (outdir/'outline.json').write_text(json.dumps(outline,indent=2,ensure_ascii=False),encoding='utf-8')
    for chapter in outline:
        scenes=[]
        for scene_no in range(1,int(spec.get('scenes_per_chapter',2))+1):
            memory=' | '.join(state['events'][-6:])[-1100:]
            prompt=(f"<INSTRUCTION>\nGenre: {spec.get('genre','fantasy')}\n"
                    f"Characters: {', '.join(names)}\nSetting: {spec.get('setting','')}\n"
                    f"Goal: {chapter['objective']}\nObstacle: {chapter['obstacle']}\n"
                    f"Outcome: {chapter['intended_outcome']}\nPremise: {spec.get('premise','')}\n"
                    f"Chapter: {chapter['number']} of {len(outline)}; phase: {chapter['stage']}\n"
                    f"Scene: {scene_no}\nPreviously established events: {memory}\n"
                    'Write a vivid, original scene with concrete actions and dialogue. Maintain continuity.\n<STORY>\n')
            text=generate(model,tok,prompt,max_new_tokens,temperature,top_k)
            if not text: text='[MODEL PRODUCED NO SCENE]'
            missing=[n for n in names if not re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)',text,re.I)]
            scenes.append(text)
            state['events'].append(f"Chapter {chapter['number']} scene {scene_no}: " + text[:240].replace('\n',' '))
            if missing:
                state.setdefault('warnings',[]).append({'chapter':chapter['number'],'scene':scene_no,'missing_characters':missing})
        content=f"# Chapter {chapter['number']}: {chapter['stage']}\n\n"+'\n\n'.join(scenes)+'\n'
        (outdir/f"chapter_{chapter['number']:02d}.md").write_text(content,encoding='utf-8')
        state['chapters'].append({'number':chapter['number'],'stage':chapter['stage'],'file':f"chapter_{chapter['number']:02d}.md"})
        (outdir/'story_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    (outdir/'novel.md').write_text('# '+state['title']+'\n\n'+'\n\n'.join((outdir/c['file']).read_text(encoding='utf-8') for c in state['chapters']),encoding='utf-8')
    print(f"Generated {len(outline)} chapters in {outdir}; warnings={len(state.get('warnings',[]))}")
    print('Warning: planning and continuity checks are software scaffolding; prose quality depends on training.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spec',default='story_spec.example.json')
    p.add_argument('--checkpoint',default='checkpoints/storyai_v14_instruction.pt')
    p.add_argument('--tokenizer',default='dataset/processed/tokenizer.json')
    p.add_argument('--outdir',default='output/v14_novel')
    p.add_argument('--max-new-tokens',type=int,default=350)
    p.add_argument('--temperature',type=float,default=.85)
    p.add_argument('--top-k',type=int,default=40)
    a=p.parse_args()
    run(json.loads(Path(a.spec).read_text(encoding='utf-8')),a.checkpoint,a.tokenizer,a.outdir,a.max_new_tokens,a.temperature,a.top_k)

if __name__=='__main__': main()
