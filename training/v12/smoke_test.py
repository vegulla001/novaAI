"""CPU-only data/model interface smoke test; never updates checkpoints."""
import argparse, hashlib, json, pathlib, torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def prompt_for(r):
 return ('<INSTRUCTION>\nWrite a '+', '.join(r['genres'])+' story as narrative prose.\n'
         +'Title: '+r['title']+'\nPremise: '+r['premise']+'\nCharacters: '+', '.join(r['characters'])+'\n'
         +'Setting: '+r['setting']+'\nChapter 1: '+r['objective']+'\n'
         +'Write only the scene.\n<STORY>\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',default='dataset/v12/pilot/train.jsonl');ap.add_argument('--tokenizer',default='tokenizer.json');ap.add_argument('--checkpoint',default='checkpoints/storyai_v8_best.pt');ap.add_argument('--skip-checkpoint',action='store_true');a=ap.parse_args()
 records=[json.loads(s) for s in pathlib.Path(a.data).read_text(encoding='utf-8').splitlines() if s.strip()]
 assert records,'No examples'
 for r in records:
  assert all(r.get(k) for k in ('title','genres','characters','premise','story','objective'))
  for name in r['characters']:
   assert name.casefold() in r['story'].casefold(),f'Missing character {name} in {r["title"]}'
 print('Schema and character grounding: OK;',len(records),'examples')
 tok=BPETokenizer.load(a.tokenizer)
 if a.skip_checkpoint:
  print('Tokenizer load OK; skipped checkpoint test');return
 try: ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 except TypeError: ckpt=torch.load(a.checkpoint,map_location='cpu')
 if 'tokenizer_sha256' in ckpt:
  digest=hashlib.sha256(pathlib.Path(a.tokenizer).read_bytes()).hexdigest()
  assert digest==ckpt['tokenizer_sha256'],'Tokenizer checksum differs from checkpoint'
 cfg=GPTConfig(**ckpt['config']);model=GPT(cfg);model.load_state_dict(ckpt['model']);model.train()
 r=records[0];prefix=tok.encode(prompt_for(r));story=tok.encode(r['story']+'\n<END_STORY>')
 ids=prefix+story
 assert len(ids)>len(prefix)+2
 block=cfg.block_size
 if len(ids)>block+1:
  # Preserve entire instruction prefix; truncate story tail only.
  assert len(prefix)<block-2,'Prompt exceeds model context'
  ids=ids[:block+1]
 x=torch.tensor([ids[:-1]],dtype=torch.long)
 y=torch.tensor([ids[1:]],dtype=torch.long)
 y[:,:len(prefix)-1]=-1
 assert torch.all(y[:,:len(prefix)-1]==-1),'Prompt targets not masked'
 assert torch.any(y!=-1),'Story targets all masked'
 model.zero_grad(set_to_none=True)
 _,loss=model(x,y)
 assert torch.isfinite(loss),'Nonfinite loss'
 loss.backward()
 assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'No finite gradients'
 print(f'CPU forward/backward OK; context={len(ids)-1}, prompt={len(prefix)}, loss={loss.item():.4f}')
 print('No optimizer step performed. No checkpoint was changed.')
if __name__=='__main__':main()
