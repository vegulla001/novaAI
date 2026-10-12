import json
import uuid
import copy
from pathlib import Path


class StoryManager:

    def __init__(self, folder='stories'):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def new(self, title, premise=''):
        story = {
            'story_id': uuid.uuid4().hex[:12],
            'title': title,
            'premise': premise,
            'characters': [],
            'plot': '',
            'chapters': [],
            'history': [],
            'undo_stack': []
        }

        self.save(story)
        return story

    def save(self, story):
        path = self.folder / (story['story_id'] + '.json')
        temp = path.with_suffix('.tmp')

        temp.write_text(
            json.dumps(story, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

        temp.replace(path)

    def load(self, story_id):
        path = self.folder / (story_id + '.json')

        return json.loads(
            path.read_text(encoding='utf-8')
        )

    def _snapshot(self, story):
        story['undo_stack'].append(
            copy.deepcopy({
                key: story[key]
                for key in ('characters', 'plot', 'chapters')
            })
        )

    def _find_character(self, story, name):
        for character in story['characters']:
            if character['name'].casefold() == name.casefold():
                return character

        return None

    def _flag_chapters(self, story, names=None):
        for chapter in story['chapters']:
            if names is None:
                chapter['needs_revision'] = True
            elif any(
                name.casefold() in {
                    character_name.casefold()
                    for character_name in chapter.get('characters', [])
                }
                for name in names
            ):
                chapter['needs_revision'] = True

    def apply(self, story, action):
        kind = action.get('action')

        # ------------------------------------------
        # UNDO
        # ------------------------------------------

        if kind == 'UNDO':
            if not story['undo_stack']:
                return 'Nothing to undo.'

            state = story['undo_stack'].pop()

            for key in ('characters', 'plot', 'chapters'):
                story[key] = state[key]

            story['history'].append({'action': 'UNDO'})
            self.save(story)

            return 'Undid the last change.'

        supported = {
            'ADD_CHARACTER',
            'UPDATE_CHARACTER',
            'ADD_RELATIONSHIP',
            'SET_PLOT',
            'ADD_CHAPTER'
        }

        if kind not in supported:
            return (
                'I could not safely understand that request. '
                'Please rephrase.'
            )

        # ------------------------------------------
        # ADD CHARACTER
        # ------------------------------------------

        if kind == 'ADD_CHARACTER':
            name = action.get('name', '').strip()
            role = action.get('role', '').strip()

            if not name or not role:
                return 'Please provide character name and role.'

            if self._find_character(story, name):
                return 'Character already exists.'

            self._snapshot(story)

            character = {
                'name': name,
                'role': role,
                'story_position': action.get('story_position'),
                'description': action.get('description', ''),
                'personality': [],
                'abilities': [],
                'relationships': []
            }

            story['characters'].append(character)

            result = f'Added {name} as {role}.'

        # ------------------------------------------
        # UPDATE CHARACTER
        # ------------------------------------------

        elif kind == 'UPDATE_CHARACTER':
            name = action.get('name', '').strip()
            role = action.get('role', '').strip()

            if not name or not role:
                return 'Please provide character name and role.'

            character = self._find_character(story, name)

            if not character:
                return 'Character not found.'

            self._snapshot(story)

            character['role'] = role

            if action.get('story_position'):
                character['story_position'] = action['story_position']

            self._flag_chapters(story, [name])

            result = (
                f'Updated {character["name"]} to {role}. '
                'Relevant chapters flagged.'
            )

        # ------------------------------------------
        # ADD RELATIONSHIP
        # ------------------------------------------

        elif kind == 'ADD_RELATIONSHIP':
            name1 = action.get('character1', '').strip()
            name2 = action.get('character2', '').strip()
            relationship = action.get('relationship', '').strip()

            if not name1 or not name2 or not relationship:
                return 'Please provide both characters and relationship.'

            character1 = self._find_character(story, name1)
            character2 = self._find_character(story, name2)

            if not character1 or not character2:
                return 'One or both characters do not exist.'

            if character1 is character2:
                return 'A character cannot have a relationship with themselves.'

            relation1 = {
                'character': character2['name'],
                'type': relationship
            }

            relation2 = {
                'character': character1['name'],
                'type': relationship
            }

            character1.setdefault('relationships', [])
            character2.setdefault('relationships', [])

            if (
                relation1 in character1['relationships']
                and relation2 in character2['relationships']
            ):
                return 'Relationship already exists.'

            self._snapshot(story)

            if relation1 not in character1['relationships']:
                character1['relationships'].append(relation1)

            if relation2 not in character2['relationships']:
                character2['relationships'].append(relation2)

            self._flag_chapters(
                story,
                [character1['name'], character2['name']]
            )

            result = (
                f'{character1["name"]} and '
                f'{character2["name"]} are now {relationship}.'
            )

        # ------------------------------------------
        # SET PLOT
        # ------------------------------------------

        elif kind == 'SET_PLOT':
            plot = action.get('plot', '').strip()

            if not plot:
                return 'Please describe the plot change.'

            self._snapshot(story)

            story['plot'] = plot

            self._flag_chapters(story)

            result = (
                'Updated plot. '
                'Existing chapters flagged for revision.'
            )

        # ------------------------------------------
        # ADD CHAPTER
        # ------------------------------------------

        elif kind == 'ADD_CHAPTER':
            title = action.get('title', '').strip()

            if not title:
                return 'Please provide a chapter title.'

            self._snapshot(story)

            story['chapters'].append({
                'title': title,
                'content': '',
                'characters': [],
                'needs_revision': True
            })

            result = f'Added chapter {title}.'

        else:
            return 'Unsupported story action.'

        # ------------------------------------------
        # SAVE
        # ------------------------------------------

        story['history'].append(copy.deepcopy(action))

        self.save(story)

        return result