import argparse
from .inference import ConversationAI
from story_engine.manager import StoryManager

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/storyai_v9_conversation.pt')
    args = p.parse_args()
    manager = StoryManager()
    ai = ConversationAI(args.checkpoint)
    print('StoryAI V9 chat. Commands: /new TITLE, /open ID, /show, /model TEXT, /quit')
    story = None
    while True:
        try:
            msg = input('You > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == '/quit':
            break
        if msg.startswith('/new '):
            story = manager.new(msg[5:].strip())
            ai.last_character = None
            print(f"StoryAI > Created {story['title']} ID: {story['story_id']}")
            continue
        if msg.startswith('/open '):
            try:
                story = manager.load(msg[6:].strip())
                ai.last_character = None
                print('StoryAI > Opened', story['title'])
            except (OSError, ValueError) as e:
                print('StoryAI >', e)
            continue
        if msg == '/show':
            print(story if story else 'No story selected')
            continue
        if msg.startswith('/model '):
            print('Model raw (diagnostic only) >', ai.model_raw(msg[7:]))
            continue
        if story is None:
            print('StoryAI > First use /new Your Story Title')
            continue
        action = ai.interpret(msg, story)
        print('Action >', action)
        result = manager.apply(story, action)
        print('StoryAI >', result)
        if action['action'] == 'UNDO':
            ai.last_character = None
        elif action['action'] == 'ADD_CHARACTER' and result.startswith('Character already exists'):
            ai.last_character = action['name']

if __name__ == '__main__':
    main()
