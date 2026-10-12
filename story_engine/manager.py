import json,uuid,copy
from pathlib import Path

class StoryManager:
    def __init__(self,folder='stories'):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
    def new(self,title,premise=''):
        s={'story_id':uuid.uuid4().hex[:12],'title':title,'premise':premise,'characters':[],'plot':'','chapters':[],'history':[],'undo_stack':[]}
        self.save(s);return s
    def save(self,story):
        path=self.folder/(story['story_id']+'.json');temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(story,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
    def load(self,story_id):
        return json.loads((self.folder/(story_id+'.json')).read_text(encoding='utf-8'))
    def apply(self,story,action):
        kind=action.get('action')
        if kind=='UNDO':
            if not story['undo_stack']: return 'Nothing to undo.'
            state=story['undo_stack'].pop()
            for key in ('characters','plot','chapters'): story[key]=state[key]
            story['history'].append({'action':'UNDO'});self.save(story);return 'Undid the last change.'
        if kind not in ('ADD_CHARACTER','UPDATE_CHARACTER','SET_PLOT','ADD_CHAPTER'):
            return 'I could not safely understand that request. Please rephrase.'
        name=action.get('name','').strip();role=action.get('role','').strip()
        if kind in ('ADD_CHARACTER','UPDATE_CHARACTER') and (not name or not role): return 'Please provide character name and role.'
        if kind=='ADD_CHARACTER' and any(c['name'].casefold()==name.casefold() for c in story['characters']):return 'Character already exists.'
        if kind=='UPDATE_CHARACTER' and not any(c['name'].casefold()==name.casefold() for c in story['characters']):return 'Character not found.'
        if kind=='SET_PLOT' and not action.get('plot'):return 'Please describe the plot change.'
        if kind=='ADD_CHAPTER' and not action.get('title'):return 'Please provide a chapter title.'
        story['undo_stack'].append(copy.deepcopy({k:story[k] for k in ('characters','plot','chapters')}))
        if kind=='ADD_CHARACTER':
            story['characters'].append({'name':name,'role':role});result=f'Added {name} as {role}.'
        elif kind=='UPDATE_CHARACTER':
            for c in story['characters']:
                if c['name'].casefold()==name.casefold(): c['role']=role
            for ch in story['chapters']:
                if name in ch.get('characters',[]):ch['needs_revision']=True
            result=f'Updated {name} to {role}. Relevant chapters flagged.'
        elif kind=='SET_PLOT':
            story['plot']=action['plot']
            for ch in story['chapters']:ch['needs_revision']=True
            result='Updated plot. Existing chapters flagged for revision.'
        else:
            story['chapters'].append({'title':action['title'],'content':'','characters':[],'needs_revision':True});result='Added chapter '+action['title']+'.'
        story['history'].append(action);self.save(story);return result
