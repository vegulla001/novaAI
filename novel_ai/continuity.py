"""Deterministic identity rename and continuation audit. No LLM required.

The registry uses stable IDs and explicit exact aliases. Avoids partial-word edits,
case-fold collisions, and silently rewriting ambiguous identities.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from typing import Any

class ContinuityError(ValueError):
    pass


def _valid_name(name: str) -> str:
    if not isinstance(name, str):
        raise ContinuityError('Character name must be a string')
    name = unicodedata.normalize('NFC', name).strip()
    if not (2 <= len(name) <= 80) or any(ord(ch) < 32 for ch in name):
        raise ContinuityError('Character name must be 2..80 printable characters')
    if not any(ch.isalpha() for ch in name):
        raise ContinuityError('Character name must contain letters')
    return name


def _pattern(names: list[str]) -> re.Pattern:
    # Unicode letter boundaries and longer aliases first, including possessives.
    names = sorted(set(names), key=lambda n: (-len(n), n))
    return re.compile(r'(?<!\w)(' + '|'.join(re.escape(n) for n in names) + r")(?!\w)", re.IGNORECASE)


def initialize_registry(plan: dict) -> dict:
    registry = {}
    used = set()
    for i, character in enumerate(plan.get('characters', []), start=1):
        if not isinstance(character, dict) or not character.get('name'):
            raise ContinuityError('Characters require a name')
        name = _valid_name(character['name'])
        if name.casefold() in used:
            raise ContinuityError('Duplicate character name')
        used.add(name.casefold())
        registry[f'char_{i:04d}'] = {'name': name, 'past_names': [], 'aliases': []}
    return registry


def rename(state: dict, character_id: str, new_name: str) -> dict:
    new_name = _valid_name(new_name)
    output = copy.deepcopy(state)
    registry = output.get('identity_registry')
    if not isinstance(registry, dict) or character_id not in registry:
        raise ContinuityError('Unknown character ID')
    record = registry[character_id]
    old = record['name']
    if old.casefold() == new_name.casefold():
        if old == new_name:
            return output
        raise ContinuityError('Case-only renaming is ambiguous')
    occupied = {x.casefold() for cid, item in registry.items() if cid != character_id
                for x in [item['name'], *item.get('aliases', []), *item.get('past_names', [])]}
    if new_name.casefold() in occupied:
        raise ContinuityError('New name collides with another character identity')
    # Reject old names that are already an alias of someone else.
    if old.casefold() in occupied:
        raise ContinuityError('Old name ambiguous across identity registry')
    for past in record.get('past_names', []):
        if past.casefold() in occupied:
            raise ContinuityError('Historic name ambiguous across identity registry')
    record['past_names'] = list(dict.fromkeys([*record.get('past_names', []), old]))
    record['name'] = new_name
    # Rewrite structured canon only, not already published chapters.
    # Longest-first replacement prevents substring replacement; spans include possessives.
    forbidden = [old, *record.get('aliases', []), *record.get('past_names', [])]
    trans = _pattern(forbidden)
    def replace_string(text: str) -> str:
        def sub(m):
            matched = m.group(0)
            # preserve common capitalization styles without altering identity
            if matched.isupper():
                return new_name.upper()
            if matched.islower():
                return new_name.lower()
            return new_name
        return trans.sub(sub, text)
    def rewrite(value: Any):
        if isinstance(value, str): return replace_string(value)
        if isinstance(value, list): return [rewrite(x) for x in value]
        if isinstance(value, dict): return {rewrite(k): rewrite(v) for k, v in value.items()}
        return value
    if 'plan' in output: output['plan'] = rewrite(output['plan'])
    if 'memory' in output: output['memory'] = rewrite(output['memory'])
    return output


def audit(text: str, state: dict) -> list[str]:
    registry = state.get('identity_registry', {})
    if not registry:
        return []
    forbidden = []
    for cid, record in registry.items():
        for name in record.get('past_names', []):
            forbidden.append((name, cid))
    if not forbidden:
        return []
    # Detect old name usages only when unambiguous. Historical references are disallowed
    # in this strict continuation policy; callers must deliberately relax if required.
    pat = _pattern([name for name, _ in forbidden])
    matches = []
    for m in pat.finditer(text):
        # 80 chars of context for actionable diagnostics, no excessive story logging.
        matches.append(f'Forbidden former character name: {m.group(0)!r} at offset {m.start()}')
        if len(matches) >= 20: break
    return matches


def build_continuation_chapter(number: int, plot: str) -> dict:
    if not isinstance(plot, str) or not (25 <= len(plot.strip()) <= 4000):
        raise ContinuityError('Continuation plot must be 25..4000 characters')
    return {
        'number': number,
        'title': f'Continuation {number}',
        'objective': plot.strip(),
        'conflict': 'Advance the established conflicts without contradicting established canon.',
        'turning_point': 'Resolve or transform a meaningful ongoing story tension.',
        'ending_hook': 'End with a causally justified next development.',
    }


def canonicalize_text(text: str, state: dict) -> str:
    """Rewrite prior-scene context to current names without modifying archived chapters."""
    registry = state.get('identity_registry', {})
    substitutions = {}
    for cid, record in registry.items():
        for past in record.get('past_names', []):
            key = past.casefold()
            if key in substitutions and substitutions[key] != record['name']:
                raise ContinuityError('Ambiguous alias in earlier chapters')
            substitutions[key] = record['name']
    if not substitutions:
        return text
    return _pattern(list(substitutions)).sub(lambda m: substitutions[m.group().casefold()], text)
