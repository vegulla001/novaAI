import json
import unittest
from copy import deepcopy
from pathlib import Path
from training.v16.reasoning_planner import solve, PlanError

SPEC = json.loads((Path(__file__).resolve().parents[1] / 'examples' / 'castle_reasoning.json').read_text(encoding='utf-8'))

class PlannerTests(unittest.TestCase):
    def test_reaches_goal(self):
        r = solve(SPEC)
        self.assertEqual(r['status'], 'solved')
        self.assertTrue(r['goal_satisfied'])
        self.assertTrue(r['final_state']['safe_route_identified'])
    def test_avoids_forbidden_shortcut(self):
        r = solve(SPEC)
        self.assertFalse(r['final_state']['gate_open'])
        self.assertNotIn('force_northern_gate', [p['action_id'] for p in r['plan']])
    def test_optimal_plan(self):
        r = solve(SPEC)
        self.assertEqual([p['action_id'] for p in r['plan']], ['scout_wall','find_side_entrance','mark_safe_route'])
    def test_unsatisfiable(self):
        s = deepcopy(SPEC)
        s['actions'] = [a for a in s['actions'] if a['id'] in ('force_northern_gate', 'listen_to_gate')]
        self.assertEqual(solve(s)['status'], 'unsatisfiable')
    def test_goal_already_true(self):
        s = deepcopy(SPEC)
        s['initial_state']['safe_route_identified'] = True
        self.assertEqual(solve(s)['plan'], [])
    def test_state_transition(self):
        r = solve(SPEC)
        self.assertFalse(r['plan'][0]['state_after']['safe_route_identified'])
        self.assertTrue(r['plan'][-1]['state_after']['safe_route_identified'])
    def test_unknown_actor(self):
        s = deepcopy(SPEC)
        s['actions'][0]['actor'] = 'stranger'
        with self.assertRaises(PlanError): solve(s)
    def test_invalid_initial_state(self):
        s = deepcopy(SPEC)
        s['initial_state']['gate_open'] = True
        with self.assertRaises(PlanError): solve(s)
    def test_changed_goal_replans(self):
        s = deepcopy(SPEC)
        s['goal'] = {'sound_detected': True}
        self.assertEqual([p['action_id'] for p in solve(s)['plan']], ['listen_to_gate'])
    def test_unreachable_goal(self):
        s = deepcopy(SPEC)
        s['goal'] = {'father_found': True}
        self.assertEqual(solve(s)['status'], 'unsatisfiable')

if __name__ == '__main__': unittest.main()
