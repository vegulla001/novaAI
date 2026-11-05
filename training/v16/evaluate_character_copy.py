#!/usr/bin/env python3
"""Compare v16 models on exact copying and simple named continuation; no output injection."""
import argparse, json, re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from training.v15.model import GPT,GPTConfig

SCENES=[
 ('castle','Arin','Elara','Arin stood outside the abandoned castle with a letter. Elara held a lantern by the northern gate.'),
 ('harbor','Mira','Jonas','Mira waited near the harbor. Jonas approached carrying a sealed map.'),
 ('library','Nila','Dev','Nila stood inside the old library. Dev heard noises behind a locked passage.'),
]
SEEDS=[42,137,2026]

def has_name(text,name):
 return bool(re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text,re.I))

def load_model(path,device):
 ckpt=torch.load(path,map_location='cpu',weights_only=False)
 model=GPT(GPTConfig(**ckpt['model_config'])).to(device)
 model.load_state_dict(ckpt['model'],strict=True)
 model.eval()
 return model

def generate(model,tok,prompt,device,seed,max_tokens,greedy,block_size):
 ids=tok.encode(prompt,add_special_tokens=False).ids
 if not ids:raise ValueError('Empty prompt token ids')
 x=torch.tensor([ids],device=device,dtype=torch.long)
 torch.manual_seed(seed)
 if device=='cuda':torch.cuda.manual_seed_all(seed)
 with torch.inference_mode():
  if greedy:
   # Explicit greedy loop, since model.generate may not support temperature=0.
   out=x
   for _ in range(max_tokens):
    logits,_=model(out[:,-block_size:])
    next_id=torch.argmax(logits[:,-1,:],dim=-1,keepdim=True)
    out=torch.cat([out,next_id],dim=1)
  else:
   out=model.generate(x,max_new_tokens=max_tokens,temperature=0.7,top_k=40)
 return tok.decode(out[0].tolist()[len(ids):],skip_special_tokens=True)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--checkpoints',nargs='+',required=True)
 p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
 p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
 p.add_argument('--output',default='reports/v16_character_copy_comparison.json')
 p.add_argument('--print-samples',action='store_true')
 a=p.parse_args()
 device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
 tok=Tokenizer.from_file(a.tokenizer)
 report={'seeds':SEEDS,'checkpoints':[]}
 for checkpoint in a.checkpoints:
  model=load_model(checkpoint,device)
  block_size=1024
  cases=[]
  for title,n1,n2,scene in SCENES:
   copy_prompt=f'Copy exactly the two character names from this story. Do not invent names.\nStory: {scene}\nCharacters: '
   scene_prompt=(f'Continue this story in one sentence using both named characters.\n'
                 f'Characters to preserve: {n1}, {n2}\nStory so far:\n{scene}\nContinuation:\n')
   for seed in SEEDS:
    copy_out=generate(model,tok,copy_prompt,device,seed,24,greedy=True,block_size=block_size)
    prose=generate(model,tok,scene_prompt,device,seed,80,greedy=False,block_size=block_size)
    # Exact first line with no surrounding prompt is the objective of the copy task.
    first_line=copy_out.splitlines()[0].strip()
    expected=f'{n1}, {n2}'
    result={'scene':title,'seed':seed,'required':[n1,n2],
      'copy_first_line':first_line,'copy_exact':first_line==expected,
      'prose_generated':prose,'prose_mentions':{n1:has_name(prose,n1),n2:has_name(prose,n2)},
      'prose_both':has_name(prose,n1) and has_name(prose,n2)}
    cases.append(result)
    print(checkpoint,title,'seed',seed,'COPY_EXACT',result['copy_exact'],'PROSE_BOTH',result['prose_both'])
    if a.print_samples:print('COPY',repr(first_line),'\nPROSE',prose[:650],'\n')
  summary={'checkpoint':checkpoint,'cases':len(cases),
    'copy_exact_count':sum(x['copy_exact'] for x in cases),
    'copy_exact_rate':sum(x['copy_exact'] for x in cases)/len(cases),
    'prose_both_count':sum(x['prose_both'] for x in cases),
    'prose_both_rate':sum(x['prose_both'] for x in cases)/len(cases),
    'cases_detail':cases}
  report['checkpoints'].append(summary)
  print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k!='cases_detail'},indent=2))
  del model
  if device=='cuda':torch.cuda.empty_cache()
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(report,indent=2),encoding='utf8')
 print('SAVED',out)
if __name__=='__main__':main()
