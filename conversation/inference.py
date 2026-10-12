"""StoryAI V9: safe interpretation for supported conversational story edits.

The from-scratch TinyGPT is loaded for experimentation, but unsupported requests
are rejected rather than trusting malformed or hallucinated model JSON.
"""
import json
import re
import torch
from .model import TinyGPT
from .train import EOS, encode, prompt

ROLE = r'(?:mage|warrior|healer|villain|ranger|knight|scholar|princess|prince|wizard|witch|hero|antagonist|protagonist|blacksmith|dragon|king|queen)'
NAME = r'[A-Z][a-zA-Z]*(?:[ -][A-Z][a-zA-Z]*)?'

class ConversationAI:
    def __init__(self, checkpoint='checkpoints/storyai_v9_conversation.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = TinyGPT(**ck['config']).to(self.device)
        self.model.load_state_dict(ck['model'])
        self.model.eval()
        self.last_character = None

    @torch.inference_mode()
    def model_raw(self, message, max_tokens=160):
        """Diagnostic only: model output is NOT executed automatically."""
        ids = encode(prompt(message))
        if len(ids) >= self.model.block_size:
            return ''
        generated = []
        for _ in range(max_tokens):
            x = torch.tensor([ids[-self.model.block_size:]], device=self.device)
            logits, _ = self.model(x)
            nxt = int(torch.argmax(logits[0, -1]).item())
            if nxt in (EOS, 10) or nxt > 255:
                break
            generated.append(nxt)
            ids.append(nxt)
        return bytes(generated).decode('utf-8', errors='replace')

    def interpret(self, message, story=None):
        msg = message.strip()
        existing = {c['name'].casefold(): c['name'] for c in (story or {}).get('characters', [])}

        if re.fullmatch(r'(?:undo|undo that(?: change)?|undo the last (?:action|change)|revert my last edit)[.! ]*', msg, re.I):
            return {'action': 'UNDO'}
        # ------------------------------------------
        # ADD CHARACTER
        # ------------------------------------------

        m = re.fullmatch(
            r'(?:add|create|introduce)\s+'
            r'(?:(?:a|an|new)\s+)*'
            r'(?:character\s+)?'
            r'(?:(?:named|called)\s+)?'
            r'(' + NAME + r')\s+'
            r'(?:who\s+is|as|is)\s+'
            r'(?:(?:a|an)\s+)?'
            r'(?:(?:powerful|mysterious|young|brave|evil)\s+)*'
            r'(' + ROLE + r')'
            r'(?:\s+and\s+(?:(?:the|a|an)\s+)?'
            r'(protagonist|antagonist|main character|supporting character)'
            r'(?:\s+of\s+the\s+story)?)?'
            r'[.! ]*',
            msg,
            re.I
        )

        if m:
            name = m.group(1).strip()
            role = m.group(2).lower()
            story_position = m.group(3)

            action = {
                'action': 'ADD_CHARACTER',
                'name': name,
                'role': role
            }

            if story_position:
                action['story_position'] = story_position.lower()

            return action
        # ------------------------------------------
        # CHARACTER RELATIONSHIPS
        # ------------------------------------------

        m = re.fullmatch(
            r'(?:make|put|place)\s+'
            r'(' + NAME + r')\s+and\s+'
            r'(' + NAME + r')\s+'
            r'(?:(?:as|in|on)\s+)?'
            r'(?:the\s+)?same\s+team'
            r'[.! ]*',
            msg,
            re.I
        )

        if m:
            name1 = m.group(1).strip()
            name2 = m.group(2).strip()

            character1 = existing.get(name1.casefold())
            character2 = existing.get(name2.casefold())

            if not character1 or not character2:
                return {
                    'action': 'UNRECOGNIZED',
                    'reason': 'One or both characters do not exist.'
                }

            if character1.casefold() == character2.casefold():
                return {
                    'action': 'UNRECOGNIZED',
                    'reason': 'Cannot create a team relationship with the same character.'
                }

            return {
                'action': 'ADD_RELATIONSHIP',
                'character1': character1,
                'character2': character2,
                'relationship': 'teammates'
            }
        # --------------------------------------------------
        # UPDATE CHARACTER
        # --------------------------------------------------

        # First handle pronoun-based follow-up instructions.
        m = re.fullmatch(
            r'(?:make|change|update|turn)\s+'
            r'(her|him|them)\s+'
            r'(?:(?:into|to be|to|as)\s+)?'
            r'(?:a\s+|an\s+)?'
            r'(' + ROLE + r')[.! ]*',
            msg,
            re.I
        )

        if m:
            role = m.group(2).lower()

            if not self.last_character:
                return {
                    'action': 'UNRECOGNIZED',
                    'reason': 'No previously mentioned character.'
                }

            return {
                'action': 'UPDATE_CHARACTER',
                'name': self.last_character,
                'role': role
            }

        # Then handle explicitly named characters.
        m = re.fullmatch(
            r'(?:make|change|update|turn)\s+'
            r'(' + NAME + r')\s+'
            r'(?:(?:into|to be|to|as)\s+)?'
            r'(?:a\s+|an\s+)?'
            r'(' + ROLE + r')[.! ]*',
            msg,
            re.I
        )

        if m:
            requested = m.group(1).strip()
            role = m.group(2).lower()

            name = existing.get(requested.casefold())

            if not name:
                return {
                    'action': 'UNRECOGNIZED',
                    'reason': f'Character {requested} does not exist.'
                }

            return {
                'action': 'UPDATE_CHARACTER',
                'name': name,
                'role': role
            }

        m = re.fullmatch(r'(?:add|create)(?: a)?(?: new)? chapter (?:called|titled|named)\s+(.+?)[.! ]*', msg, re.I)
        if m:
            return {'action': 'ADD_CHAPTER', 'title': m.group(1).strip(' ."\'')}

        m = re.fullmatch(r'(?:change|update|set|make)\s+(?:the\s+)?(?:plot|storyline)\s+(?:so|to|about how|to be)\s+(.+?)[.! ]*', msg, re.I)
        if m:
            return {'action': 'SET_PLOT', 'plot': m.group(1).strip()}

        return {'action': 'UNRECOGNIZED', 'reason': 'Unsupported phrasing. No story changes made.'}
