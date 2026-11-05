#!/usr/bin/env python3
"""Compare raw model-only character mentions, without any character-name insertion."""
import argparse
import json
import re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from training.v15.model import GPT, GPTConfig

SCENES = [
    ('castle', 'The rain had not stopped for three days. Arin stood outside the abandoned castle, holding a letter from his missing father. Behind him, Elara approached with a lantern. "Arin," she whispered, "there is something moving beyond the northern gate." Arin unfolded the letter. The final warning was written in his father\'s handwriting: Never open the northern gate.', ['Arin','Elara'], 'northern gate'),
    ('harbor', 'At dawn, Mira waited at the harbor for her brother Jonas. He had promised to return before the storm. The fishing boats were empty, and a red scarf lay on the wet pier. Mira recognized it as Jonas\'s scarf. She called his name, but nobody answered.', ['Mira','Jonas'], 'harbor'),
    ('library', 'The library was silent when Nila found the torn map. Her friend Dev had hidden it beneath a history book. They had agreed not to enter the old tunnel until sunrise. Outside, the clock struck midnight. Nila heard footsteps in the corridor.', ['Nila','Dev'], 'tunnel'),
]
HEADER = 'Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\n'

def count(s, name):
    return len(re.findall(r'(?<!\w)'+re.escape(name)+r'(?!\w)',s,flags=re.I))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--tokens',type=int,default=180)
    p.add_argument('--temperature',type=float,default=0.7)
    p.add_argument('--top-k',type=int,default=30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--output',default='reports/v16_character_learned_comparison.json')
    a=p.parse_args()
    dev=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    tok=Tokenizer.from_file(a.tokenizer)
    results=[]
    for cp in a.checkpoints:
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        model=GPT(GPTConfig(**ck['model_config'])).to(dev)
        model.load_state_dict(ck['model'],strict=True); model.eval()
        for title,scene,names,setting in SCENES:
            # Exactly the character-labelled format used by the new training dataset.
            prompt=HEADER+f'Characters to preserve: {", ".join(names)}\n\nStory so far:\n{scene}\n\nContinuation:\n'
            ids=tok.encode(prompt).ids
            if len(ids)+a.tokens>int(ck['model_config']['block_size']):
                raise ValueError('Prompt + generation exceeds model context limit')
            torch.manual_seed(a.seed)
            if dev=='cuda':torch.cuda.manual_seed_all(a.seed)
            x=torch.tensor([ids],dtype=torch.long,device=dev)
            with torch.inference_mode():
                output=model.generate(x,max_new_tokens=a.tokens,temperature=a.temperature,top_k=a.top_k)
            completion=tok.decode(output[0,len(ids):].tolist(),skip_special_tokens=True)
            mentions={name:count(completion,name) for name in names}
            result={'checkpoint':cp,'scene':title,'model_only_mentions':mentions,
                    'both_characters_present':all(v>0 for v in mentions.values()),
                    'setting_mentioned':setting.lower() in completion.lower(),
                    'json_contamination':bool(re.search(r'"(?:observed_next_text|context_evidence)"\s*:',completion)),
                    'generated_text':completion}
            results.append(result)
            print(f'\n===== {cp} | {title} =====\n{completion}\n')
            print('MODEL_ONLY_MENTIONS:',mentions,'BOTH:',result['both_characters_present'], 'SETTING:',result['setting_mentioned'])
        del model,ck
        if dev=='cuda':torch.cuda.empty_cache()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SAVED',out)
if __name__=='__main__': main()
