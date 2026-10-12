"""Neural intent + character-level slot tagging; no regex-based command parsing."""
import torch
from .model import NLUModel,encode
from .dataset import INTENTS,TAGS

class StoryNLU:
    def __init__(self,checkpoint='checkpoints/storyai_v10_nlu.pt',threshold=.70):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.model=NLUModel(hidden=ck['hidden']).to(self.device)
        self.model.load_state_dict(ck['model']);self.model.eval();self.threshold=threshold
    @torch.inference_mode()
    def predict(self,message):
        seq=encode(message)
        if not seq or len(message.encode('utf-8'))>len(seq):return {'action':'UNRECOGNIZED','reason':'Message empty or too long'}
        x=torch.tensor([seq],device=self.device);lengths=torch.tensor([len(seq)],device=self.device)
        il,tl=self.model(x,lengths)
        probs=il.softmax(-1)[0];confidence,idx=probs.max(0)
        intent=INTENTS[idx.item()];score=confidence.item()
        if score<self.threshold or intent=='UNKNOWN':return {'action':'UNRECOGNIZED','reason':f'Low confidence or unknown intent ({score:.2f})'}
        predicted=tl.argmax(-1)[0].tolist()
        spans=[];current=None
        for i,tid in enumerate(predicted+[0]):
            tag=TAGS[tid]
            if tag.startswith('B-') or (tag.startswith('I-') and (current is None or current[0]!=tag[2:])):
                if current:spans.append((current[0],bytes(seq[current[1]:i]).decode('utf-8',errors='replace')))
                current=(tag[2:],i)
            elif tag=='O':
                if current:spans.append((current[0],bytes(seq[current[1]:i]).decode('utf-8',errors='replace')));current=None
            elif current and current[0]!=tag[2:]:
                spans.append((current[0],bytes(seq[current[1]:i]).decode('utf-8',errors='replace')));current=(tag[2:],i)
        slots={}
        for label,value in spans:slots.setdefault(label,[]).append(value.strip())
        return {'intent':intent,'confidence':round(score,3),'slots':slots,'text':message}

    def interpret(self,message,story=None,last_character=None):
        pred=self.predict(message)
        if 'intent' not in pred:return pred
        intent=pred['intent'];s=pred['slots']
        def one(key):return s.get(key,[None])[0]
        def unknown(reason):return {'action':'UNRECOGNIZED','reason':reason,'prediction':pred}
        existing={c['name'].casefold():c['name'] for c in (story or {}).get('characters',[])}
        def known(name):return existing.get(name.casefold()) if name else None
        if intent=='UNDO':return {'action':'UNDO'}
        if intent=='ADD_CHARACTER':
            if not one('CHAR') or not one('ROLE'):return unknown('Missing character name or role')
            a={'action':intent,'name':one('CHAR'),'role':one('ROLE')}
            if one('POSITION'):a['story_position']=one('POSITION')
            return a
        if intent=='UPDATE_CHARACTER':
            name=one('CHAR') or (last_character if one('REF') else None)
            if not name or not one('ROLE'):return unknown('Missing character or role')
            if not known(name):return unknown(f'Character not found: {name}')
            return {'action':intent,'name':known(name),'role':one('ROLE')}
        if intent=='ADD_RELATIONSHIP':
            chars=s.get('CHAR',[])
            if len(chars)!=2:return unknown('Need two character names')
            a,b=known(chars[0]),known(chars[1])
            if not a or not b or a==b:return unknown('Both distinct characters must exist')
            # Only confidently map team phrasing; otherwise require labeled relation.
            relation=one('RELATION')
            if relation is None and 'same team' in message.casefold():relation='teammates'
            if relation is None:return unknown('Relationship type not recognized')
            return {'action':intent,'character1':a,'character2':b,'relationship':relation}
        if intent=='SET_PLOT':
            if not one('PLOT'):return unknown('Missing plot text')
            return {'action':intent,'plot':one('PLOT')}
        if intent=='ADD_CHAPTER':
            if not one('TITLE'):return unknown('Missing chapter title')
            return {'action':intent,'title':one('TITLE')}
        return unknown('Unsupported action')
