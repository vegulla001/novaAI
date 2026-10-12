import argparse

from .inference import ConversationAI
from story_engine.manager import StoryManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/storyai_v9_conversation.pt'
    )
    args = parser.parse_args()

    manager = StoryManager()
    ai = ConversationAI(args.checkpoint)

    print(
        'StoryAI V9 chat. Commands: '
        '/new TITLE, /open ID, /show, /model TEXT, /quit'
    )

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

            print(
                f"StoryAI > Created {story['title']} "
                f"ID: {story['story_id']}"
            )
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
            print(
                'Model raw (diagnostic only) >',
                ai.model_raw(msg[7:])
            )
            continue

        if story is None:
            print('StoryAI > First use /new Your Story Title')
            continue

        action = ai.interpret(msg, story)

        print('Action >', action)

        result = manager.apply(story, action)

        print('StoryAI >', result)

        # ------------------------------------------
        # UPDATE CONVERSATION MEMORY
        # ------------------------------------------

        if action.get('action') == 'UNDO':
            ai.last_character = None

        elif action.get('action') in (
            'ADD_CHARACTER',
            'UPDATE_CHARACTER'
        ):
            name = action.get('name')

            if name:
                try:
                    saved_story = manager.load(story['story_id'])

                    character_exists = any(
                        character['name'].casefold() == name.casefold()
                        for character in saved_story.get('characters', [])
                    )

                    if character_exists:
                        ai.last_character = name
                        story = saved_story

                except (OSError, ValueError) as error:
                    print(
                        'StoryAI > Could not refresh character memory:',
                        error
                    )


if __name__ == '__main__':
    main()