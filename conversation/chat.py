import argparse
from .inference import ConversationAI
from story_engine.manager import StoryManager

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/storyai_v9_conversation.pt');a=p.parse_args()
    manager=StoryManager();ai=ConversationAI(a.checkpoint)
    print('StoryAI V9 experimental chat. Type /new TITLE, /open ID, /show, /quit.')
    story=None
    while True:
        try: msg=input('You > ').strip()
        except (EOFError,KeyboardInterrupt):break
        if not msg:continue
        if msg=='/quit':break
        if msg.startswith('/new '):
            story=manager.new(msg[5:].strip());print('StoryAI > Created',story['title'],'ID:',story['story_id']);continue
        if msg.startswith('/open '):
            try:story=manager.load(msg[6:].strip());print('StoryAI > Opened',story['title'])
            except (OSError,ValueError) as e:print('StoryAI >',e)
            continue
        if msg=='/show':print(story if story else 'No story selected');continue
        if story is None:print('StoryAI > First use /new Your Story Title');continue
        action=ai.interpret(msg)
        print('Action >',action)
        print('StoryAI >',manager.apply(story,action))
if __name__=='__main__':main()
