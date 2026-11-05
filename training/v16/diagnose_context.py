"""Read-only Story AI v16 checkpoint/data context-conditioning diagnostic.

Never trains, updates, or writes checkpoint weights. Needs local project with
training.v15.model, tokenizer, and checkpoint. No GPU required.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import re
from pathlib import Path


def check_sft_rows(rows, max_examples=100, block_size=1024):
    """Inspect causal label alignment and amount of prompt surviving crop."""
    counters = dict(examined=0, bad_lengths=0, no_supervision=0,
                    supervised_at_first_token=0, need_crop=0,
                    crop_loses_prefix=0, boundary_missing_after_crop=0,
                    prompt_tokens_before_crop_sum=0, prompt_tokens_after_crop_sum=0)
    examples = []
    for i, row in enumerate(rows[:max_examples]):
        ids, labels = list(row['input_ids']), list(row['labels'])
        counters['examined'] += 1
        if len(ids) != len(labels):
            counters['bad_lengths'] += 1
            continue
        supervised = [j for j,v in enumerate(labels) if int(v) != -100]
        if not supervised:
            counters['no_supervision'] += 1
            continue
        first = supervised[0]
        counters['prompt_tokens_before_crop_sum'] += first
        if first == 0:
            counters['supervised_at_first_token'] += 1
        start = 0
        if len(ids) > block_size+1:
            counters['need_crop'] += 1
            start = max(0,first-max(1,block_size//2))
            start = min(start, len(ids)-(block_size+1))
            if start:
                counters['crop_loses_prefix'] += 1
        trimmed = labels[start:start+block_size+1]
        remain = next((j for j, v in enumerate(trimmed) if int(v)!=-100), -1)
        if remain == -1:
            counters['boundary_missing_after_crop'] += 1
        counters['prompt_tokens_after_crop_sum'] += max(0, remain)
        if len(examples) < 5:
            examples.append({'row':i,'length':len(ids),'first_supervised_idx':first,
                             'crop_start':start,'prompt_tokens_retained':max(0,remain),
                             'first_target_equals_input': int(labels[first])==int(ids[first])})
    n = max(1, counters['examined']-counters['bad_lengths']-counters['no_supervision'])
    counters['average_prompt_tokens_before_crop'] = round(counters.pop('prompt_tokens_before_crop_sum')/n,2)
    counters['average_prompt_tokens_after_crop'] = round(counters.pop('prompt_tokens_after_crop_sum')/n,2)
    return {'counts': counters,'examples':examples,
            'interpretation':'target-label equality is expected when labels are aligned to input_ids; trainer shifts labels[1:] against input_ids[:-1]'}


def token_nll(model, prompt_ids, completion_ids, block_size, device):
    """Mean teacher-forced NLL for completion with either GPT forward contract.

    Handles models returning [batch, sequence, vocab] *or* optimized
    inference logits [batch, 1, vocab] (final position only).
    Does not assume the model implements training logits without targets.
    """
    import torch
    import torch.nn.functional as F

    prompt_ids = list(prompt_ids)
    completion_ids = list(completion_ids)
    if not prompt_ids or not completion_ids:
        raise ValueError('Prompt and completion must both contain tokens')
    total = len(prompt_ids) + len(completion_ids)
    if total > block_size:
        raise ValueError(f'Prompt + target is {total} tokens > context {block_size}; shorten it')

    # Full teacher-forced sequence. At position t the model predicts t+1.
    all_ids = prompt_ids + completion_ids
    x = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(completion_ids, dtype=torch.long, device=device)
    first_prediction_index = len(prompt_ids) - 1

    with torch.inference_mode():
        output = model(x)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError(f'Unexpected forward logits shape: {tuple(logits.shape)}')

        # Standard training-style forward returning all time steps.
        if logits.shape[1] == x.shape[1]:
            pred = logits[0, first_prediction_index:first_prediction_index + len(completion_ids)]
            return float(F.cross_entropy(pred.float(), labels, reduction='mean').item())

        # GPT.generate-oriented forward returns only the final position.
        # Score each next completion token with a growing context.
        if logits.shape[1] != 1:
            raise RuntimeError(f'Unsupported logits time dimension {logits.shape[1]} '
                               f'for input time dimension {x.shape[1]}')

        losses = []
        for i, next_token in enumerate(completion_ids):
            if i == len(completion_ids) - 1:
                next_logits = logits[0, -1]  # already computed last prefix
            else:
                current = torch.tensor([all_ids[:len(prompt_ids) + i]],
                                       dtype=torch.long, device=device)
                next_out = model(current)
                next_logits = (next_out[0] if isinstance(next_out, (tuple, list))
                               else next_out)[0, -1]
            # Prefix is all_ids[:prompt_len + i] for token i.
            # We must never reuse the full-sequence last logits for its first token.
            losses.append(F.cross_entropy(next_logits.float().unsqueeze(0),
                                          torch.tensor([next_token], device=device),
                                          reduction='mean'))

        # Replace the last logits reuse above: final full-sequence logits
        # predicts the last completion token, because x omits that token.
        return float(torch.stack(losses).mean().item())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--dataset',default='dataset/v16_character_dynamic_sft/train_sft.pt',help='Optional SFT .pt file for alignment checks')
    p.add_argument('--max-rows',type=int,default=100)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',default='reports/v16_context_diagnostic.json')
    a=p.parse_args()
    import torch
    from tokenizers import Tokenizer
    from training.v15.model import GPT,GPTConfig

    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)  # trusted locally created checkpoint only
    cfg=ckpt['model_config']
    model=GPT(GPTConfig(**cfg)).to(a.device)
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    tok=Tokenizer.from_file(a.tokenizer)
    block=int(cfg['block_size'])
    report={'checkpoint':a.checkpoint,'model_config':cfg,
            'checkpoint_step':ckpt.get('step'), 'checks':{}}

    # Source inspection is advisory; actual attention behavior must be checked with model tests.
    try:
        import training.v15.model as source
        source_text=inspect.getsource(source)
        report['checks']['source_hints']={
            'module_file':str(Path(inspect.getfile(source)).resolve()),
            'has_tril_mask_string': bool(re.search(r'tril|is_causal|scaled_dot_product_attention|causal',source_text,re.I)),
            'note':'Presence/absence of mask keywords does not prove attention correctness. Inspect the local model.py manually.'}
    except (OSError,TypeError):
        report['checks']['source_hints']={'warning':'source could not be inspected'}

    # The completion string is *identical* across both prompts. NLL improvement
    # with the matching name is evidence of using context (not proof of coherence).
    examples=[
        ('Arin','Elara','Arin', 'Elara'),
        ('Mira','Jonas','Mira','Jonas'),
        ('Nila','Dev','Nila','Dev'),
    ]
    tests=[]
    for a_name,b_name,_,_ in examples:
        for name in (a_name,b_name):
            other = b_name if name == a_name else a_name
            alt = 'Caden' if name != 'Caden' else 'Evan'
            def prefix(n):
                return ('Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\n'
                        f'Characters to preserve: {n}, {other}\n\n'
                        'Story so far:\n'
                        f'{n} stood by the northern gate. {other} carried a lantern. '
                        f'{n} held a letter and listened to the wind.\n\nContinuation:\n')
            completion=f'{name} looked toward the gate.'
            ids_yes=tok.encode(prefix(name)).ids
            ids_no=tok.encode(prefix(alt)).ids
            answer=tok.encode(completion).ids
            if not ids_yes or not ids_no or not answer:
                continue
            yes=token_nll(model,ids_yes,answer,block,a.device)
            no=token_nll(model,ids_no,answer,block,a.device)
            tests.append({'target_name':name,'wrong_prefix_name':alt,'matching_context_nll':round(yes,5),
                          'mismatched_context_nll':round(no,5),
                          'context_advantage_nats':round(no-yes,5),
                          'interpretation':'Positive favors matching context. Near zero/negative suggests limited or contrary conditioning for this task.'})
    report['checks']['context_sensitivity']=tests

    dataset_path=Path(a.dataset)
    if dataset_path.is_file():
        rows=torch.load(str(dataset_path),map_location='cpu',weights_only=True)
        if not isinstance(rows,list):
            raise ValueError('Expected dataset to be a list of dicts')
        report['checks']['sft_alignment']=check_sft_rows(rows,a.max_rows,block)
    else:
        report['checks']['sft_alignment']={'skipped':f'{dataset_path} not found'}

    report['conclusion_caveats']=[
        'These checks cannot prove novel-writing ability or identify every architecture bug.',
        'A small positive context advantage does not imply reliable scene adherence.',
        'If advantages are weak, inspect training.v15.model and original pretraining before paying for further training.',
        'No optimizer update or checkpoint modification was performed.'
    ]
    target=Path(a.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(report['checks'],indent=2,ensure_ascii=False))
    print('SAVED',target)

if __name__=='__main__':
    main()
