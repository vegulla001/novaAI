#!/usr/bin/env python3
"""Prepare Novelist narrative-continuation SFT for StoryAI V15. No tokenizer/checkpoint changes."""
import argparse, collections, hashlib, json, random, re
from pathlib import Path
import torch
from tokenizers import Tokenizer

EXPECTED='ab2daa146b2261c18bf4f8f3122136be30100a230e675ba7d70783a1fc29090d'
SENTENCE=re.compile(r'(?<=[.!?])\s+(?=["“‘A-Z0-9])|\n\s*\n+')
BAD=re.compile(r'(?im)^\s*(?:chapter\s+\d+|scene\s+\d+|act\s+[ivx]+|(?:outline|summary|premise|analysis|reasoning|thought process)\s*:|\d+[.)]\s+(?:beat|step|goal|obstacle))\b')

def split_text(text, tok, max_tokens):
    """Prefer sentence boundaries; split oversized sentences by tokens without dropping text."""
    pieces=[]
    for paragraph in re.split(r'\n\s*\n',text):
        for sentence in SENTENCE.split(paragraph.strip()):
            sentence=sentence.strip()
            if sentence:
                ids=tok.encode(sentence,add_special_tokens=False).ids
                for start in range(0,len(ids),max_tokens):
                    chunk=ids[start:start+max_tokens]
                    if chunk: pieces.append(chunk)
    chunks=[]; current=[]
    for piece in pieces:
        if len(current)+len(piece)>max_tokens and current:
            chunks.append(current); current=[]
        current.extend(piece)
    if current: chunks.append(current)
    return chunks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('dataset/v15/novelist_raw/english_narrative.jsonl'))
    p.add_argument('--tokenizer',type=Path,default=Path('dataset/v15/tokenizer.json'))
    p.add_argument('--output-dir',type=Path,default=Path('dataset/v15/novelist_continuation_sft'))
    p.add_argument('--block-size',type=int,default=1024)
    p.add_argument('--validation-percent',type=int,default=5)
    p.add_argument('--max-records',type=int,default=0,help='0=all')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    if not a.input.is_file(): raise FileNotFoundError(a.input)
    if hashlib.sha256(a.tokenizer.read_bytes()).hexdigest()!=EXPECTED: raise RuntimeError('Tokenizer hash mismatch; aborting')
    if not 1<=a.validation_percent<=25: raise ValueError('validation-percent must be 1..25')
    tok=Tokenizer.from_file(str(a.tokenizer)); bos=tok.token_to_id('<bos>'); eos=tok.token_to_id('<eos>')
    if bos is None or eos is None: raise RuntimeError('Missing BOS/EOS')
    prefix='Continue the following fiction scene naturally, preserving its characters, point of view, and events.\n\nStory so far:\n'
    suffix='\n\nContinuation:\n'
    prefix_ids=tok.encode(prefix,add_special_tokens=False).ids
    suffix_ids=tok.encode(suffix,add_special_tokens=False).ids
    # Every example contains a genuine preceding prose context and a genuine following prose segment.
    # Reserve prompt+context+continuation within block_size+1, avoiding trainer truncation.
    context_budget=180
    answer_budget=a.block_size-1-len(prefix_ids)-len(suffix_ids)-context_budget-1
    if answer_budget<128: raise RuntimeError('Context window too small for this prompt')
    data={'train':[],'val':[]}; stats=collections.Counter(); unique=set(); books={'train':set(),'val':set()}; genres=collections.Counter()
    with a.input.open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if not line.strip():continue
            if a.max_records and stats['rows']>=a.max_records:break
            stats['rows']+=1
            try: row=json.loads(line)
            except json.JSONDecodeError: stats['invalid_json']+=1;continue
            content=row.get('text','')
            if not isinstance(content,str) or len(content.split())<200 or BAD.search(content):
                stats['invalid_or_outline']+=1;continue
            content=content.strip()
            digest=hashlib.sha256(re.sub(r'\s+',' ',content).casefold().encode()).hexdigest()
            if digest in unique:stats['exact_duplicate']+=1;continue
            unique.add(digest)
            book=row.get('book_id') or row.get('story_id')
            if not book:
                # Unknown group cannot be safely split by book. Keep it in training only.
                split='train';stats['missing_group_train_only']+=1
            else:
                h=int(hashlib.sha256(str(book).encode()).hexdigest()[:12],16)%100
                split='val' if h<a.validation_percent else 'train'
                books[split].add(str(book))
            genre=str(row.get('genre') or 'unspecified'); genres[genre]+=1
            # Tokenize scene once and use contiguous, non-overlapping target spans.
            ids=tok.encode(content,add_special_tokens=False).ids
            if len(ids)<context_budget+128:stats['too_short_tokens']+=1;continue
            pos=context_budget
            while pos<len(ids):
                target=ids[pos:pos+answer_budget]
                if len(target)<96:stats['short_tail_dropped']+=1;break
                context=ids[max(0,pos-context_budget):pos]
                input_ids=[bos]+prefix_ids+context+suffix_ids+target+[eos]
                labels=[-100]*(1+len(prefix_ids)+len(context)+len(suffix_ids))+target+[eos]
                if len(input_ids)>a.block_size+1:raise AssertionError('Unexpected truncation')
                if len(input_ids)!=len(labels):raise AssertionError('Labels misaligned')
                data[split].append({'input_ids':input_ids,'labels':labels})
                pos+=len(target)
            stats['accepted_scenes']+=1
    overlap=books['train'] & books['val']
    if overlap:raise RuntimeError(f'Book leakage detected: {len(overlap)}')
    report={'source':str(a.input),'tokenizer_sha256':EXPECTED,'block_size':a.block_size,'train_examples':len(data['train']),'val_examples':len(data['val']),'train_books':len(books['train']),'val_books':len(books['val']),'overlap_books':len(overlap),'answer_budget_tokens':answer_budget,'context_budget_tokens':context_budget,'counts':dict(stats),'top_genres':genres.most_common(20),'caveats':['Synthetic scenes may contain repetitive or inconsistent writing; manual quality review remains necessary.','Split is by book_id, falling back to story_id; rows with neither go only to train.','This is continuation SFT, not instruction-to-generate-complete-scene SFT.']}
    print(json.dumps(report,indent=2))
    if not data['train'] or not data['val']:raise RuntimeError('Empty train or validation split')
    if a.apply:
        if a.output_dir.exists():raise FileExistsError(f'Output directory exists: {a.output_dir}')
        a.output_dir.mkdir(parents=True)
        for split,rows in data.items():
            torch.save(rows,a.output_dir/f'{split}_sft.pt')
        (a.output_dir/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Saved to',a.output_dir)
    else:print('PREVIEW ONLY. Use --apply to save.')
if __name__=='__main__':main()
