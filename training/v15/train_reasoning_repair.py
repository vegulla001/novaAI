#!/usr/bin/env python3
"""StoryAI V15 focused instruction-learning run from existing SFT weights.

Trains real GPT weights (no symbolic planner, no pretrained third-party weights).
Uses separate story/reasoning batches, monitors held-out greedy generation,
and saves checkpoints under a new output directory.
"""
import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch

from training.v15.model import GPT, GPTConfig
from training.v15.train import get_sft_batch


def hash_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_data(path, vocab):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, list) or not data:
        raise ValueError(f'Empty/invalid dataset: {path}')
    for rec in data:
        ids, labels = rec['input_ids'], rec['labels']
        if len(ids) < 2 or len(ids) != len(labels):
            raise ValueError(f'Invalid example in {path}')
        if not any(int(z) >= 0 for z in labels[1:]):
            raise ValueError(f'No supervised answer in {path}')
        if any(not (0 <= int(z) < vocab) for z in ids):
            raise ValueError(f'Invalid token IDs in {path}')
        if any((int(z) not in (-100, -1)) and not (0 <= int(z) < vocab) for z in labels):
            raise ValueError(f'Invalid label IDs in {path}')
        if isinstance(labels, torch.Tensor) and (labels == -1).any():
            labels = labels.clone()
            labels[labels == -1] = -100
            rec['labels'] = labels
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='dataset/v15_prose_expanded')
    p.add_argument('--story-dir', default='dataset/v15_prose_expanded/continuation_sft')
    p.add_argument('--reasoning-dir', default='dataset/v15_prose_expanded/reasoning_v2_sft')
    p.add_argument('--init-from', required=True)
    p.add_argument('--out-dir', default='checkpoints/v15_reasoning_repair')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--reasoning-weight', type=float, default=0.8)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-batches', type=int, default=10)
    p.add_argument('--greedy-cases', type=int, default=20)
    p.add_argument('--greedy-tokens', type=int, default=24)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', choices=['cuda','cpu','auto'], default='cuda')
    p.add_argument('--eval-only', action='store_true', help='Evaluate checkpoint and exit; no output files')
    p.add_argument('--max-story-loss-increase', type=float, default=0.02, help='Maximum allowed increase in held-out story loss when selecting best checkpoint')
    a = p.parse_args()
    if min(a.steps,a.batch_size,a.grad_accum,a.eval_every,a.eval_batches,a.greedy_cases,a.greedy_tokens) < 1:
        p.error('Steps, batches, and evaluation settings must be positive')
    if not 0 < a.reasoning_weight <= 1 or not 0 < a.min_lr <= a.lr:
        p.error('Invalid reasoning weight or learning rate')
    if a.max_story_loss_increase < 0:
        p.error('Story loss tolerance cannot be negative')
    if a.warmup < 0:
        p.error('Warmup cannot be negative')
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    data_dir = Path(a.data_dir)
    report = json.loads((data_dir/'build_report.json').read_text(encoding='utf-8-sig'))
    tok_hash = hash_file(data_dir/'tokenizer.json')
    if tok_hash != report['tokenizer_sha256']:
        raise ValueError('Tokenizer hash mismatch with report')
    # Load only trusted checkpoints created by this project.
    ckpt = torch.load(a.init_from, map_location='cpu', weights_only=False)
    if ckpt['tokenizer_sha256'] != tok_hash or ckpt.get('mode') not in ('sft','pretrain'):
        raise ValueError('Checkpoint/tokenizer or mode mismatch')
    cfg = GPTConfig(**ckpt['model_config'])
    if cfg.vocab_size != int(report['vocab_size']):
        raise ValueError('Checkpoint vocabulary mismatch')
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(data_dir/'tokenizer.json'))
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt

    rows = {}
    for task, directory in (('reasoning',a.reasoning_dir),('story',a.story_dir)):
        d=Path(directory)
        rows[task]={split:load_data(d/f'{split}_sft.pt',cfg.vocab_size) for split in ('train','val')}
    print('DATA',json.dumps({k:{s:len(z) for s,z in v.items()} for k,v in rows.items()}),flush=True)

    # Fixed held-out samples and exact tokenizer-ID evaluation avoid prompt-template differences.
    sample_rng = random.Random(90210)
    sample_ids = sample_rng.sample(range(len(rows['reasoning']['val'])),min(a.greedy_cases,len(rows['reasoning']['val'])))
    samples = [rows['reasoning']['val'][i] for i in sample_ids]
    for rec in samples:
        lab = list(rec['labels'])
        boundary=next((i for i,v in enumerate(lab) if int(v)>=0),None)
        if boundary is None or boundary < 1:
            raise ValueError('Invalid held-out prompt/response boundary')
        if boundary>cfg.block_size:
            raise ValueError('Evaluation prompt exceeds context size')

    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.1)
    bf16 = device=='cuda' and torch.cuda.is_bf16_supported()
    autocast=lambda: torch.autocast(device_type='cuda',dtype=torch.bfloat16 if bf16 else torch.float16) if device=='cuda' else nullcontext()
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and not bf16)

    def batch(task,split):
        return get_sft_batch(rows[task][split],a.batch_size,cfg.block_size,device)

    @torch.no_grad()
    def heldout():
        model.eval()
        rng_state=random.getstate()
        torch_state=torch.get_rng_state()
        random.seed(1729)
        torch.manual_seed(1729)
        result={}
        try:
            for task in ('reasoning','story'):
                losses=[]
                for _ in range(a.eval_batches):
                    x,y=batch(task,'val')
                    with autocast():
                        _,loss=model(x,y)
                    losses.append(float(loss))
                result[task+'_loss']=sum(losses)/len(losses)
            # Greedy exact-prefix accuracy. A strict metric: small 24-token answers count fully,
            # longer answers count only on the first 24 tokens.
            matched=0
            examples=[]
            for rec in samples:
                ids=[int(v) for v in rec['input_ids']]
                labels=[int(v) for v in rec['labels']]
                boundary=next(i for i,v in enumerate(labels) if v>=0)
                prompt=ids[:boundary]
                target=ids[boundary:boundary+a.greedy_tokens]
                seq=torch.tensor([prompt],dtype=torch.long,device=device)
                generated=model.generate(seq,max_new_tokens=len(target),top_k=1)[0].tolist()[len(prompt):]
                ok=(generated==target)
                matched+=int(ok)
                if len(examples)<5:
                    examples.append({'expected':tok.decode(target,skip_special_tokens=False)[:200],
                                     'generated':tok.decode(generated,skip_special_tokens=False)[:200],
                                     'correct_prefix':ok})
            result['greedy_prefix_correct']=matched
            result['greedy_prefix_total']=len(samples)
            result['greedy_prefix_accuracy']=matched/len(samples)
            result['examples']=examples
        finally:
            random.setstate(rng_state)
            torch.set_rng_state(torch_state)
            model.train()
        return result

    out=Path(a.out_dir)
    if not a.eval_only:
        if out.exists() and any(out.glob('*.pt')):
            raise RuntimeError(f'Refusing to overwrite existing checkpoints in {out}')
        out.mkdir(parents=True,exist_ok=True)

    def save(step,stats,name):
        state={'model':model.state_dict(),'model_config':vars(cfg).copy(),
               'mode':'sft','step':step,'tokenizer_sha256':tok_hash,
               'metrics':stats,'objective':{'reasoning':a.reasoning_weight,'story':1-a.reasoning_weight}}
        path=out/name
        torch.save(state,path)
        print('SAVED',path,flush=True)

    baseline=heldout()
    print('BASELINE',json.dumps(baseline),flush=True)
    if a.eval_only:
        return
    best_accuracy=baseline['greedy_prefix_accuracy']
    best_reason_loss=baseline['reasoning_loss']
    max_story_loss = baseline['story_loss'] + a.max_story_loss_increase

    for step in range(1,a.steps+1):
        if a.warmup and step<=a.warmup:
            lr=a.lr*step/a.warmup
        else:
            progress=max(0,min(1,(step-a.warmup)/max(1,a.steps-a.warmup)))
            lr=a.min_lr+0.5*(a.lr-a.min_lr)*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        train_stats={'reasoning':0.,'story':0.}
        for _ in range(a.grad_accum):
            x_r,y_r=batch('reasoning','train')
            x_s,y_s=batch('story','train')
            with autocast():
                _,loss_r=model(x_r,y_r)
                _,loss_s=model(x_s,y_s)
                loss=a.reasoning_weight*loss_r+(1-a.reasoning_weight)*loss_s
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            train_stats['reasoning']+=float(loss_r.detach())/a.grad_accum
            train_stats['story']+=float(loss_s.detach())/a.grad_accum
            scaler.scale(loss/a.grad_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()
        if step==1 or step%10==0:
            print('TRAIN',step,'lr',f'{lr:.3g}',json.dumps(train_stats),flush=True)
        if step%a.eval_every==0 or step==a.steps:
            stats=heldout()
            print('EVAL',step,json.dumps(stats),flush=True)
            save(step,stats,'sft_latest.pt')
            story_ok = stats['story_loss'] <= max_story_loss
            better=story_ok and (stats['greedy_prefix_accuracy']>best_accuracy or
                    (stats['greedy_prefix_accuracy']==best_accuracy and stats['reasoning_loss']<best_reason_loss))
            if not story_ok:
                print('BEST_SKIPPED_STORY_REGRESSION', stats['story_loss'], 'limit', max_story_loss, flush=True)
            if better:
                best_accuracy=stats['greedy_prefix_accuracy']
                best_reason_loss=stats['reasoning_loss']
                save(step,stats,'sft_best.pt')
    print('DONE. Do not claim reasoning success unless generated answers match held-out targets.',flush=True)

if __name__=='__main__':
    main()
