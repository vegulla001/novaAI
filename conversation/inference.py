import json,torch
from .model import TinyGPT
from .train import EOS,encode,prompt

class ConversationAI:
    def __init__(self,checkpoint='checkpoints/storyai_v9_conversation.pt'):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=TinyGPT(**ck['config']).to(self.device);self.model.load_state_dict(ck['model']);self.model.eval()
    @torch.inference_mode()
    def interpret(self,message,max_tokens=160):
        ids=encode(prompt(message));generated=[]
        if len(ids)>=self.model.block_size: raise ValueError('User prompt too long for model')
        for _ in range(max_tokens):
            x=torch.tensor([ids[-self.model.block_size:]],device=self.device)
            logits,_=self.model(x)
            nxt=int(torch.argmax(logits[0,-1]).item())
            if nxt==EOS or nxt==10: break
            if nxt>255: break
            generated.append(nxt);ids.append(nxt)
        raw=bytes(generated).decode('utf-8',errors='replace')
        try: return json.loads(raw)
        except json.JSONDecodeError: return {'action':'UNRECOGNIZED','raw':raw}
