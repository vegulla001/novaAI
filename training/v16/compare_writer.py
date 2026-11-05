#!/usr/bin/env python3
"""Compare StoryAI V15/V16 checkpoints on identical, SFT-formatted novel prompts.
This is a diagnostic; human assessment is required for narrative coherence.
"""
import argparse
import json
import re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig

SCENES = [
    ('castle', '''Chapter One: The Forgotten Kingdom\n\nThe rain had not stopped for three days. Arin stood outside the abandoned castle, holding a letter from his missing father. Behind him, Elara approached with a lantern.\n\n"Arin," she whispered, "there is something moving beyond the northern gate."\n\nArin unfolded the letter. The final warning was written in his father's handwriting: Never open the northern gate.''', ['Arin','Elara','gate']),
    ('harbor', '''At dawn, Mira waited at the harbor for her brother Jonas. He had promised to return before the storm. The fishing boats were empty, and a red scarf lay on the wet pier. Mira recognized it as Jonas's scarf. She called his name, but nobody answered.''', ['Mira','Jonas','harbor']),
    ('library', '''The library was silent when Nila found the torn map. Her friend Dev had hidden it beneath a history book. They had agreed not to enter the old tunnel until sunrise. Outside, the clock struck midnight. Nila heard footsteps in the corridor.''', ['Nila','Dev','tunnel']),
]
HEADER='Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\nStory so far:\n'

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--checkpoints', nargs='+', required=True)
    a.add_argument('--tokenizer', default='dataset/v15_prose_expanded/tokenizer.json')
    a.add_argument('--tokens', type=int, default=220)
    a.add_argument('--temperature', type=float, default=0.7)
    a.add_argument('--top-k', type=int, default=30)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    a.add_argument('--output', default='reports/storyai_writer_comparison.json')
    args=a.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    tok=Tokenizer.from_file(args.tokenizer)
    outcomes=[]
    for cp in args.checkpoints:
        if not Path(cp).is_file():
            print('SKIPPING missing:',cp,flush=True);continue
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        model=GPT(GPTConfig(**ck['model_config'])).to(device)
        model.load_state_dict(ck['model'],strict=True);model.eval()
        max_context=ck['model_config']['block_size']
        for title,scene,names in SCENES:
            prompt=HEADER+scene+'\n\nContinuation:\n'
            ids=tok.encode(prompt).ids
            if len(ids)+args.tokens>max_context:
                raise ValueError(f'Context overflow: prompt {len(ids)} + output {args.tokens} > block_size {max_context}')
            torch.manual_seed(args.seed)
            if device=='cuda':torch.cuda.manual_seed_all(args.seed)
            x=torch.tensor([ids],dtype=torch.long,device=device)
            with torch.inference_mode():
                y=model.generate(x,max_new_tokens=args.tokens,temperature=args.temperature,top_k=args.top_k)
            completion=tok.decode(y[0].tolist()[len(ids):],skip_special_tokens=True)
            mentions={name:bool(re.search(r'\b'+re.escape(name)+r'\b',completion,re.I)) for name in names}
            row={'checkpoint':cp,'scene':title,'prompt_tokens':len(ids),'expected_entities':names,
                 'entity_mentions':mentions,'json_contamination':bool(re.search(r'"(?:observed_next_text|context_evidence)"\s*:',completion)),
                 'continuation':completion}
            outcomes.append(row)
            print(f'\n===== {cp} | {title} =====\n{completion}\n',flush=True)
            print('ENTITY_MENTIONS:',mentions,'JSON_CONTAMINATION:',row['json_contamination'],flush=True)
        del model,ck
        if device=='cuda':torch.cuda.empty_cache()
    if not outcomes:raise ValueError('No checkpoints evaluated')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(outcomes,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',out)

if __name__=='__main__':main()
