"""v16: join verified symbolic planning to registry-backed scene composition.

No learning is performed. The planner chooses among author-supplied actions;
registry beats are author-supplied linguistic renderings of those actions.
The optional model only suggests short, validated atmospheric details.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from typing import Any

from training.v16.reasoning_planner import solve, PlanError, _match
from training.v16.registry_writer import generate_story, DetailGenerator, StoryError

class IntegrationError(ValueError):
    pass


def _validate_bridge(planner: dict, bridge: dict) -> None:
    if not isinstance(bridge, dict):
        raise IntegrationError('bridge must be an object')
    chars = bridge.get('characters')
    if not isinstance(chars, dict) or not chars:
        raise IntegrationError('bridge requires a character registry')
    if set(chars) != set(planner['characters']):
        raise IntegrationError('planner and bridge character IDs must match')
    for cid in chars:
        if chars[cid].get('name') != planner['characters'][cid]['name']:
            raise IntegrationError('character name mismatch for ' + cid)
    templates = bridge.get('action_beats')
    if not isinstance(templates, dict):
        raise IntegrationError('action_beats must map action IDs to beat templates')
    if not isinstance(bridge.get('location'), str) or not bridge['location'].strip():
        raise IntegrationError('bridge requires a location')
    for action in planner['actions']:
        aid = action['id']
        if aid not in templates:
            raise IntegrationError(f'missing beat mapping for action {aid}')
        beat = templates[aid]
        if not isinstance(beat, dict) or beat.get('actor') != action['actor']:
            raise IntegrationError(f'beat actor mismatch for {aid}')
        if 'set_state' in beat or 'id' in beat:
            raise IntegrationError(f'beat {aid} cannot define id or state changes; planner owns state')
    extra = set(templates) - {a['id'] for a in planner['actions']}
    if extra:
        raise IntegrationError('unknown action beat mappings: ' + ', '.join(sorted(extra)))
    required = bridge.get('required_characters', list(chars))
    if not isinstance(required, list) or not set(required) <= set(chars):
        raise IntegrationError('invalid required_characters')


def compose(planner: dict[str, Any], bridge: dict[str, Any], *, model=None,
            previous_state: dict[str, bool] | None = None, detail_tokens: int = 32) -> dict[str, Any]:
    """Return explicit status, proof trace, and story; fail closed on unachievable goal.

    State is committed only when the complete plan and all beat validations pass.
    """
    planner = copy.deepcopy(planner)
    bridge = copy.deepcopy(bridge)
    # Planner performs full validation including action IDs and state shapes.
    from training.v16.reasoning_planner import validate
    validate(planner)
    _validate_bridge(planner, bridge)
    if previous_state is not None:
        if not isinstance(previous_state, dict) or not all(isinstance(k,str) and isinstance(v,bool) for k,v in previous_state.items()):
            raise IntegrationError('previous state must be boolean fact mapping')
        unknown = set(previous_state) - set(planner['initial_state'])
        if unknown:
            raise IntegrationError('unknown persisted state facts: ' + ', '.join(sorted(unknown)))
        # Previous state is authoritative; do not silently revert facts to scenario defaults.
        planner['initial_state'].update(previous_state)
    result = solve(planner)
    if result['status'] != 'solved':
        return {'status': result['status'], 'plan': result, 'scene': None,
                'state_committed': False, 'reason': result.get('reason', 'No valid plan found')}
    if not result['plan']:
        return {'status': 'solved_no_actions', 'plan': result, 'scene': None,
                'state_committed': False, 'reason': 'Goal already reached; no new story events required.'}
    actions = {a['id']: a for a in planner['actions']}
    beats = []
    for number, step in enumerate(result['plan'], 1):
        aid = step['action_id']
        template = copy.deepcopy(bridge['action_beats'][aid])
        template['id'] = f'planned_{number:03d}_{aid}'
        template['set_state'] = copy.deepcopy(actions[aid]['effects'])
        beats.append(template)
    story = {'title': bridge.get('title', planner.get('title', 'Untitled')),
             'location': bridge['location'], 'characters': bridge['characters'],
             'required_characters': bridge.get('required_characters', list(bridge['characters'])),
             'forbidden_phrases': bridge.get('forbidden_phrases', []),
             'initial_state': copy.deepcopy(planner['initial_state']), 'beats': beats}
    scene = generate_story(story, model=model, detail_tokens=detail_tokens)
    if scene['state'] != result['final_state']:
        raise IntegrationError('scene/planner final-state mismatch: refusing to commit')
    if not _match(scene['state'], planner['goal']) or not _match(scene['state'], planner['invariants']):
        raise IntegrationError('scene violates verified goal or invariants')
    return {'status': 'solved', 'plan': result, 'scene': scene,
            'state_committed': True,
            'provenance': {'planning': 'symbolic_search_on_supplied_actions',
                           'plot_and_dialogue': 'supplied_beat_templates',
                           'optional_model': 'atmospheric_detail_only',
                           'model_generated_full_scene': False}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', required=True)
    p.add_argument('--bridge', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--tokenizer', default='dataset/v15_prose_expanded/tokenizer.json')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--detail-tokens', type=int, default=32)
    p.add_argument('--prompt-style', choices=['continuation', 'instruction'], default='continuation')
    p.add_argument('--state-in', help='previous boolean fact-state JSON')
    p.add_argument('--state-out', default='reports/v16_plan_to_scene_state.json')
    p.add_argument('--output', default='reports/v16_plan_to_scene.json')
    p.add_argument('--text-output', default='reports/v16_plan_to_scene.txt')
    args = p.parse_args()
    planner = json.loads(Path(args.scenario).read_text(encoding='utf8'))
    bridge = json.loads(Path(args.bridge).read_text(encoding='utf8'))
    state = json.loads(Path(args.state_in).read_text(encoding='utf8')) if args.state_in else None
    model = DetailGenerator(args.checkpoint, args.tokenizer, args.device, args.seed, args.prompt_style) if args.checkpoint else None
    result = compose(planner, bridge, model=model, previous_state=state, detail_tokens=args.detail_tokens)
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('STATUS:', result['status'])
    print('COMMITTED:', result['state_committed'])
    if result['scene'] is not None:
        scene = result['scene']
        txt = Path(args.text_output)
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(scene['text'] + '\n', encoding='utf8')
        state_out = Path(args.state_out)
        state_out.parent.mkdir(parents=True, exist_ok=True)
        state_out.write_text(json.dumps(scene['state'], indent=2) + '\n', encoding='utf8')
        print('\n' + scene['text'])
        print('\nCHARACTERS:', scene['character_mentions'])
        print('MODEL DETAILS ACCEPTED:', scene['ai_detail_count'])
        print('STATE SAVED:', state_out)
    else:
        print('REASON:', result['reason'])
        print('No story or state was written.')
    print('REPORT:', report_path)

if __name__ == '__main__':
    main()
