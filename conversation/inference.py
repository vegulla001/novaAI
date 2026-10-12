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

        m = re.fullmatch(r'(?:add|create|introduce)(?: a| an| new)?(?: character)?(?: named| called)?\s+(' + NAME + r')\s+(?:who is|as|,?\s*a|,?\s*an)\s+(?:a\s+|an\s+)?(?:powerful\s+|mysterious\s+|young\s+|brave\s+|evil\s+)*(' + ROLE + r')[.! ]*', msg, re.I)
        if m:
            name, role = m.group(1).strip(), m.group(2).lower()
            self.last_character = name
            return {'action': 'ADD_CHARACTER', 'name': name, 'role': role}

        # Additional supported natural wording: 'Add a character named Lyra who is a powerful mage.'
        m = re.fullmatch(r'(?:add|create|introduce)\s+(?:a\s+|an\s+)?character\s+(?:named|called)\s+(' + NAME + r')\s+who\s+is\s+(?:a\s+|an\s+)?(?:\w+\s+)*?(' + ROLE + r')[.! ]*', msg, re.I)
        if m:
            name, role = m.group(1), m.group(2).lower()
            self.last_character = name
            return {'action': 'ADD_CHARACTER', 'name': name, 'role': role}
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
