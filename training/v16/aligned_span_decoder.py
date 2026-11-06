"""Story AI v16: learned, tokenizer-aligned source-span answer decoder.

An intentionally narrow, testable ANSWER decoder (not a novel writing engine).
Learns to rank all source-word spans, then generates vocabulary tokens from a
mixture of the base GPT and a full-word copy distribution. No gold answer or
benchmark choice is supplied during inference. Backbone stays frozen.
"""
import argparse
import copy
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from training.v16.token_aligned_pointer import scene_slots
from training.v16.entity_span_probe import load_cases, paired_subset, verify_disjoint
from training.v16.native_pointer import hidden


class SpanCopyHead(nn.Module):
    def __init__(self, d, rank=96):
        super().__init__()
        self.query = nn.Linear(d, rank, bias=False)
        self.key = nn.Linear(d, rank, bias=False)
        self.gate = nn.Linear(d, 1)
        self.scale = rank ** -.5
        nn.init.constant_(self.gate.bias, -2.)

    def scores(self, query, keys):
        return (self.key(keys) @ self.query(query)) * self.scale


def source_record(row, tok, context):
    encoded = tok.encode(row['prompt'])
    slots = scene_slots(row['prompt'], row['scene'], encoded.offsets, tok)
    if len(encoded.ids) + 32 >= context:
        raise ValueError('Prompt too long for decoder evaluation')
    if any(not s.output_ids for s in slots):
        raise ValueError('Empty tokenized source span')
    return {'row': row, 'prompt_ids': encoded.ids, 'slots': slots,
            'answer_ids': tuple(tok.encode(' ' + row['answer']).ids)}


def gold_span_loss(scores, slots, answer):
    """Marginalize all source occurrences matching target word (train only)."""
    positives = [i for i,s in enumerate(slots) if s.text.casefold() == answer.casefold()]
    if not positives:
        raise ValueError(f'Gold span absent from source: {answer}')
    logp = F.log_softmax(scores.float(), dim=0)
    return -torch.logsumexp(logp[positives], 0)


def aligned_copy_probs(scores, slots, prefix, eos_id, vocab_size):
    """Whole-span prefix trie, including explicit EOS after a copied word.

    Prefix is the exact generated answer-token prefix. Scores are over source
    spans, not benchmark choices. All source mentions participate in scoring.
    """
    prefix = tuple(prefix)
    eligible = []
    for j,s in enumerate(slots):
        ids = s.output_ids
        if len(prefix) <= len(ids) and ids[:len(prefix)] == prefix:
            nxt = ids[len(prefix)] if len(prefix) < len(ids) else eos_id
            eligible.append((j,nxt))
    copy_p = scores.new_zeros(vocab_size, dtype=torch.float32)
    if not eligible:
        return copy_p, False
    indices = torch.tensor([i for i,_ in eligible], dtype=torch.long, device=scores.device)
    token_indices = torch.tensor([t for _,t in eligible], dtype=torch.long, device=scores.device)
    p = F.softmax(scores.float()[indices], dim=0)
    copy_p.scatter_add_(0,token_indices,p)
    return copy_p, True


def mixed_probs(head, query, vocab_logits, scores, slots, prefix, eos_id):
    vocab_p = F.softmax(vocab_logits.float(), -1)
    copy_p, available = aligned_copy_probs(scores, slots, prefix, eos_id, vocab_p.shape[-1])
    gate = torch.sigmoid(head.gate(query).float()).reshape(())
    if not available:
        return vocab_p, gate, False
    return gate * vocab_p + (1. - gate) * copy_p, gate, True


def prepare_frozen_features(model, rec, eos_id):
    """Cache frozen GPT features for teacher-forced answer + EOS."""
    answer = rec['answer_ids'] + (eos_id,)
    ids = rec['prompt_ids'] + list(answer[:-1])
    if len(ids)>model.config.block_size:
        raise ValueError('Context exceeded')
    with torch.no_grad():
        seq=torch.tensor([ids], dtype=torch.long)
        h=hidden(model,seq)[0].detach()
        logits=model.lm_head(h).detach()
    keys=torch.stack([h[s.source_position] for s in rec['slots']])
    query=h[len(rec['prompt_ids'])-1]
    start=len(rec['prompt_ids'])-1
    return dict(query=query, keys=keys, future_h=h[start:start+len(answer)],
                future_logits=logits[start:start+len(answer)], answer=answer, rec=rec)


def training_loss(head, features, eos_id, gate_weight=.25, span_weight=1.):
    rec=features['rec']; slots=rec['slots']; answer=features['answer']
    scores=head.scores(features['query'],features['keys'])
    span=gold_span_loss(scores,slots,rec['row']['answer'])
    nll=[]; gate_losses=[]
    for i,target in enumerate(answer):
        probs, gate, available=mixed_probs(head,features['future_h'][i],features['future_logits'][i],
                                           scores,slots,answer[:i],eos_id)
        nll.append(-torch.log(probs[target].clamp_min(1e-12)))
        if available:
            # For answer tokens and end marker, prefer the source-span branch.
            gate_losses.append(F.binary_cross_entropy(gate, gate.new_zeros(())))
    return torch.stack(nll).mean() + span_weight*span + gate_weight*torch.stack(gate_losses).mean()


@torch.no_grad()
def decode(model, head, rec, tok, eos_id, max_tokens=24):
    """Autoregressive next-token mixture; emits text, no post-hoc gold substitution."""
    model.eval();head.eval()
    ids=list(rec['prompt_ids'])
    src_positions=[s.source_position for s in rec['slots']]
    sequence=torch.tensor([ids],dtype=torch.long)
    h=hidden(model,sequence)[0]
    source_h=torch.stack([h[i] for i in src_positions])
    scores=head.scores(h[-1],source_h)
    result=[]; gate_values=[]
    for _ in range(max_tokens):
        if len(ids)>=model.config.block_size:break
        h=hidden(model,torch.tensor([ids],dtype=torch.long))[0]
        logits=model.lm_head(h[-1:])[0]
        p,gate,_=mixed_probs(head,h[-1],logits,scores,rec['slots'],result,eos_id)
        next_id=int(p.argmax())
        gate_values.append(float(gate))
        if next_id==eos_id:
            return tok.decode(result,skip_special_tokens=False).strip(), True, sum(gate_values)/len(gate_values)
        ids.append(next_id);result.append(next_id)
    return tok.decode(result,skip_special_tokens=False).strip(),False,sum(gate_values)/max(1,len(gate_values))


def evaluate(model,head,records,tok,eos_id,max_tokens):
    rows=[];pairs=defaultdict(list)
    for rec in records:
        ans,stop,gate=decode(model,head,rec,tok,eos_id,max_tokens)
        row=rec['row'];correct=(ans==row['answer'])
        item={'id':row['id'],'skill':row['skill'],'pair_id':row['pair_id'],
              'expected':row['answer'],'generated':ans,'correct':correct,'stopped':stop,'gate':gate}
        rows.append(item);pairs[row['pair_id']].append(correct)
    if any(len(v)!=2 for v in pairs.values()): raise ValueError('Invalid pairs')
    skills={}
    for skill in sorted({r['skill'] for r in rows}):
        sub=[r for r in rows if r['skill']==skill]
        skills[skill]=f"{sum(r['correct'] for r in sub)}/{len(sub)}"
    return {'accuracy':sum(r['correct'] for r in rows)/len(rows),
            'paired':sum(all(v) for v in pairs.values())/len(pairs),
            'stopped':sum(r['stopped'] for r in rows)/len(rows),
            'by_skill':skills},rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--data-dir',default='dataset/v16_reasoning_repair_01')
    ap.add_argument('--tokenizer',default='dataset/v15_prose_expanded/tokenizer.json')
    ap.add_argument('--train-pairs-per-skill',type=int,default=24)
    ap.add_argument('--val-pairs-per-skill',type=int,default=8)
    ap.add_argument('--steps',type=int,default=80)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--eval-every',type=int,default=20)
    ap.add_argument('--max-tokens',type=int,default=24)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--save',default='checkpoints/v16_aligned_span_01/decoder.pt')
    ap.add_argument('--out',default='reports/v16_aligned_span_01.json')
    a=ap.parse_args()
    if min(a.train_pairs_per_skill,a.val_pairs_per_skill,a.steps,a.batch_size,a.eval_every,a.max_tokens,a.threads)<1:
        ap.error('All count options must be positive')
    from tokenizers import Tokenizer
    from training.v15.model import GPT,GPTConfig
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed)
    tok=Tokenizer.from_file(str(a.tokenizer))
    eos_id=tok.token_to_id('<eos>')
    if eos_id is None:raise ValueError('Existing tokenizer lacks <eos>; no vocab change permitted')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    digest=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    if ck.get('tokenizer_sha256')!=digest:raise ValueError('Checkpoint tokenizer mismatch')
    model=GPT(GPTConfig(**ck['model_config']));model.load_state_dict(ck['model'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    all_train=load_cases(Path(a.data_dir)/'train_cases.jsonl')
    all_val=load_cases(Path(a.data_dir)/'val_cases.jsonl')
    verify_disjoint(all_train,all_val)
    train=[source_record(r,tok,model.config.block_size) for r in paired_subset(all_train,a.train_pairs_per_skill,a.seed)]
    val=[source_record(r,tok,model.config.block_size) for r in paired_subset(all_val,a.val_pairs_per_skill,a.seed+1)]
    for rec in train+val:
        if not any(s.text.casefold()==rec['row']['answer'].casefold() for s in rec['slots']):
            raise ValueError('Answer not in source; cannot train a source-copy model')
    print('CACHED TRAIN CASES:',len(train),'VAL CASES:',len(val),flush=True)
    cache=[prepare_frozen_features(model,r,eos_id) for r in train]
    head=SpanCopyHead(ck['model_config']['n_embd'])
    optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr)
    best_metric=(-1.,-1.);best_state=None;best_step=None
    history=[]
    base,base_rows=evaluate(model,head,val,tok,eos_id,a.max_tokens)
    print('STEP 0',json.dumps(base),flush=True)
    history.append(dict(step=0,**base))
    for step in range(1,a.steps+1):
        head.train();optimizer.zero_grad(set_to_none=True)
        batch=random.choices(cache,k=a.batch_size)
        losses=[]
        for features in batch:
            l=training_loss(head,features,eos_id)
            (l/a.batch_size).backward();losses.append(float(l.detach()))
        nn.utils.clip_grad_norm_(head.parameters(),1.)
        optimizer.step()
        if step%a.eval_every==0 or step==a.steps:
            score,examples=evaluate(model,head,val,tok,eos_id,a.max_tokens)
            print('STEP',step,'LOSS',round(sum(losses)/len(losses),4),json.dumps(score),flush=True)
            history.append(dict(step=step,train_loss=sum(losses)/len(losses),**score))
            metric=(score['paired'],score['accuracy'])
            if metric>best_metric:
                best_metric=metric;best_state=copy.deepcopy(head.state_dict());best_step=step
    head.load_state_dict(best_state)
    final,examples=evaluate(model,head,val,tok,eos_id,a.max_tokens)
    from training.v16.native_pointer import stress_cases
    independent=[source_record(r,tok,model.config.block_size) for r in stress_cases()]
    stress_score,stress_examples=evaluate(model,head,independent,tok,eos_id,a.max_tokens)
    print('INDEPENDENT 3+ ENTITY STRESS',json.dumps(stress_score),flush=True)
    for r in examples[:12]: print('RAW',r,flush=True)
    save=Path(a.save);out=Path(a.out)
    if save.exists() or out.exists():raise FileExistsError('Refusing to overwrite output')
    save.parent.mkdir(parents=True,exist_ok=True);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'head':head.state_dict(),'source_checkpoint':a.checkpoint,'model_config':ck['model_config'],
                'tokenizer_sha256':digest,'step':best_step,'eos_id':eos_id},save)
    out.write_text(json.dumps({'baseline':base,'selected':final,'selected_step':best_step,
        'history':history,'examples':examples,'stress':stress_score,'stress_examples':stress_examples,'limitations':[
        'This is an answer-span decoder: it does not yet perform free novel generation.',
        'Source candidates are automatically discovered word spans, not benchmark choices.',
        'Validation selection is optimistic; test on a separately held-out story distribution.',
        'The original GPT backbone remains frozen and its prose deficiencies remain.']},indent=2),encoding='utf-8')
    print('SAVED',save,'REPORT',out,flush=True)

if __name__=='__main__': main()
