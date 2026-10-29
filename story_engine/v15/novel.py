"""V15 scene controller: explicit beats, bounded continuity, honest validation.

No network, API, or training required. Plan-only mode works without Torch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def _string(value):
    return str(value or '').strip()


def make_outline(spec, outline_file=None):
    """Explicit chapters/scenes take precedence; text outline maps one line to a chapter."""
    chapters = spec.get('chapters')
    lines = []
    if outline_file:
        lines = [s.strip() for s in Path(outline_file).read_text(encoding='utf-8').splitlines() if s.strip()]
        if not lines:
            raise ValueError('Outline file is empty')
    if isinstance(chapters, list):
        if lines:
            raise ValueError('Use either spec.chapters[] or --outline-file, not both')
        raw = chapters
    else:
        count = int(chapters or len(lines) or 0)
        if not 1 <= count <= 40:
            raise ValueError('Supply 1..40 chapters or an outline file')
        if lines and len(lines) != count:
            raise ValueError(f'Outline has {len(lines)} beats, but chapters={count}. Refusing to repeat beats.')
        if not lines:
            raise ValueError('No chapter-specific plot. Supply chapters[] or --outline-file. Refusing generic repeated goals.')
        raw = [{'title': f'Chapter {i+1}', 'scenes': [{'beat': beat}]} for i, beat in enumerate(lines)]

    result = []
    for ci, chapter in enumerate(raw, 1):
        if isinstance(chapter, str):
            chapter = {'scenes': [{'beat': chapter}]}
        scenes = chapter.get('scenes', [])
        if not scenes:
            raise ValueError(f'Chapter {ci} has no scenes')
        structured = []
        for si, scene in enumerate(scenes, 1):
            if isinstance(scene, str):
                scene = {'beat': scene}
            beat = _string(scene.get('beat') or scene.get('event') or scene.get('goal'))
            if not beat:
                raise ValueError(f'Chapter {ci}, scene {si} needs a specific beat')
            structured.append({'number':si, 'beat':beat,
                'goal':_string(scene.get('goal') or beat),
                'obstacle':_string(scene.get('obstacle')),
                'outcome':_string(scene.get('outcome')),
                'required_terms':[str(x) for x in scene.get('required_terms', [])]})
        result.append({'number':ci,'title':_string(chapter.get('title')) or f'Chapter {ci}',
                       'scenes':structured})
    return result


def build_prompt(spec, chapter, scene, accepted_events, *, format='plain'):
    cast = []
    for char in spec.get('characters', []):
        if isinstance(char, dict):
            cast.append(f"{char.get('name','Unknown')}: {char.get('role','character')}; {char.get('personality','')}")
        else:
            cast.append(str(char))
    # Keep memory bounded and only use accepted text; no invented summaries.
    memory = '\n'.join(accepted_events[-3:]) or 'No prior accepted scenes.'
    common = (f"Title: {_string(spec.get('title'))}\n"
              f"Genre: {_string(spec.get('genre')) or 'fiction'}\n"
              f"Premise: {_string(spec.get('premise'))}\n"
              f"Setting: {_string(spec.get('setting'))}\n"
              f"Characters: {' | '.join(cast)}\n"
              f"Chapter {chapter['number']}: {chapter['title']}\n"
              f"Scene {scene['number']}\n"
              f"This scene's event: {scene['beat']}\n"
              f"Goal: {scene['goal']}\nObstacle: {scene['obstacle']}\n"
              f"Intended outcome: {scene['outcome']}\n"
              f"Previously accepted events (excerpts; not verified summaries): {memory}\n")
    if format == 'sft':
        # This format is NOT equivalent to the old continuation SFT; retraining would be needed.
        return ('Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\n'
                'Story so far:\n'+common+'\nContinuation:\n')
    if format == 'plain':
        return common + 'Write the next concrete fictional scene in narrative prose, continuing from these facts:\n'
    raise ValueError('Unknown prompt format')


def validate(text, scene, *, min_words=35):
    """Heuristic quality gate, not semantic proof of event compliance."""
    issues=[]
    words=re.findall(r"[\w']+", text.lower())
    if len(words) < min_words:
        issues.append('too_short')
    if len(words) >= 35 and len(set(words))/len(words) < .25:
        issues.append('low_vocabulary_diversity')
    for term in scene.get('required_terms', []):
        if not re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text,re.I):
            issues.append('missing_required_term:'+term)
    sentences=[re.sub(r'\W+',' ',s.lower()).strip() for s in re.split(r'[.!?]+',text) if len(s.split())>=5]
    if sentences and len(sentences)-len(set(sentences))>=2:
        issues.append('repeated_sentences')
    return issues


def load_model(checkpoint, tokenizer_path, device=None):
    import torch
    from tokenizers import Tokenizer
    from training.v15.model import GPT, GPTConfig
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    actual=hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    if ck.get('tokenizer_sha256') != actual:
        raise ValueError('Checkpoint/tokenizer SHA256 mismatch')
    cfg=GPTConfig(**ck.get('model_config', ck.get('config', {})))
    tok=Tokenizer.from_file(str(tokenizer_path))
    if tok.get_vocab_size()!=cfg.vocab_size:
        raise ValueError('Vocabulary mismatch')
    model=GPT(cfg)
    model.load_state_dict(ck['model'],strict=True)
    model=model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu')).eval()
    return model,tok


def generate(model,tok,prompt,*,max_new_tokens=200,temperature=.75,top_k=40,seed=42):
    import torch
    torch.manual_seed(seed)
    device=next(model.parameters()).device
    ids=tok.encode(prompt).ids
    if not ids:
        raise ValueError('Prompt produced no tokens')
    # Grounding gets priority; fail rather than silently remove chapter and memory.
    reserved=min(max_new_tokens,model.config.block_size//3)
    if len(ids)>model.config.block_size-reserved:
        raise ValueError(f'Prompt is {len(ids)} tokens; shorten memory/beat to leave {reserved} output slots')
    eos=tok.token_to_id('<eos>')
    output=[]
    with torch.inference_mode():
        for _ in range(min(max_new_tokens,model.config.block_size-len(ids))):
            idx=torch.tensor([ids+output],dtype=torch.long,device=device)
            logits,_=model(idx)
            # Supports GPT.forward inference shape [B, 1, vocab] and [B,T,vocab].
            scores=logits[0,-1,:].float()/max(float(temperature),0.05)
            if top_k>0:
                threshold=torch.topk(scores,min(top_k,scores.numel())).values[-1]
                scores[scores<threshold]=-float('inf')
            next_id=int(torch.multinomial(torch.softmax(scores,dim=-1),1))
            if eos is not None and next_id==eos:
                break
            output.append(next_id)
    return tok.decode(output,skip_special_tokens=True).strip()


def run(spec,*,outdir,outline_file=None,checkpoint=None,tokenizer=None,plan_only=False,
        prompt_format='plain',max_new_tokens=200,temperature=.75,top_k=40,seed=42):
    outline=make_outline(spec,outline_file)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    (out/'outline.json').write_text(json.dumps(outline,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'story_spec.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False),encoding='utf-8')
    model=tok=None
    if not plan_only:
        if not checkpoint or not tokenizer:
            raise ValueError('Checkpoint and tokenizer are required without --plan-only')
        model,tok=load_model(checkpoint,tokenizer)
    events=[];failures=[];chapters=[];prompt_manifest=[]
    for chapter in outline:
        scenes=[]
        for scene in chapter['scenes']:
            prompt=build_prompt(spec,chapter,scene,events,format=prompt_format)
            prompt_manifest.append({'chapter':chapter['number'],'scene':scene['number'],
                                    'beat':scene['beat'],'prompt':prompt})
            if plan_only: continue
            text=generate(model,tok,prompt,max_new_tokens=max_new_tokens,
                          temperature=temperature,top_k=top_k,
                          seed=seed+chapter['number']*1000+scene['number'])
            # Preserve every raw generation, including failures, for debugging.
            raw_dir = out / 'raw_scenes'
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"chapter_{chapter['number']:02d}_scene_{scene['number']:02d}.txt"
            raw_file.write_text(text + '\n', encoding='utf-8')
            issues=validate(text,scene)
            if issues:
                failures.append({'chapter':chapter['number'],'scene':scene['number'],
                                 'issues':issues,'raw_file':str(raw_file),
                                 'generated_words':len(text.split()),
                                 'preview':text[:250]})
                scenes.append('[REJECTED SCENE: '+', '.join(issues)+']')
            else:
                scenes.append(text)
                # Only accepted text enters memory. Explicitly excerpt, not auto-fabricated event facts.
                events.append(f"Ch{chapter['number']} scene{scene['number']}: "+' '.join(text.split())[:200])
        if not plan_only:
            name=f"chapter_{chapter['number']:02d}.md"
            (out/name).write_text('# '+chapter['title']+'\n\n'+'\n\n'.join(scenes)+'\n',encoding='utf-8')
            chapters.append(name)
            (out/'story_state.json').write_text(json.dumps({'accepted_excerpts':events,'failures':failures},indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'prompt_manifest.json').write_text(json.dumps(prompt_manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if not plan_only:
        (out/'novel.md').write_text('# '+_string(spec.get('title'))+'\n\n'+'\n\n'.join((out/n).read_text(encoding='utf-8') for n in chapters),encoding='utf-8')
    report={'mode':'plan_only' if plan_only else 'generation','scenes_planned':len(prompt_manifest),
            'scenes_rejected':len(failures),'failures':failures,
            'warning':'Heuristic checks do not verify plot comprehension; weak pretrained models may still ignore prompts.'}
    (out/'quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',default='story_spec.example.json')
    p.add_argument('--outline-file',default=None)
    p.add_argument('--outdir',default='output/v15_plotted')
    p.add_argument('--checkpoint',default='checkpoints/v15_pretrain_experiment/pretrain_latest.pt')
    p.add_argument('--tokenizer',default='dataset/v15/tokenizer.json')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--prompt-format',choices=['plain','sft'],default='plain')
    p.add_argument('--max-new-tokens',type=int,default=200)
    p.add_argument('--temperature',type=float,default=.75)
    p.add_argument('--top-k',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    spec=json.loads(Path(a.spec).read_text(encoding='utf-8'))
    print(json.dumps(run(spec,outdir=a.outdir,outline_file=a.outline_file,
                         checkpoint=a.checkpoint,tokenizer=a.tokenizer,plan_only=a.plan_only,
                         prompt_format=a.prompt_format,max_new_tokens=a.max_new_tokens,
                         temperature=a.temperature,top_k=a.top_k,seed=a.seed),indent=2))

if __name__=='__main__':main()
