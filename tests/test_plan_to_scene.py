import copy
import json
import unittest
from pathlib import Path

from training.v16.plan_to_scene import compose, IntegrationError

ROOT = Path(__file__).resolve().parents[1]

class TestPlanToScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT/'examples/castle_reasoning.json').read_text())
        cls.bridge = json.loads((ROOT/'examples/castle_plan_to_scene.json').read_text())

    def test_happy_path_and_state(self):
        r=compose(self.spec,self.bridge)
        self.assertEqual(r['status'],'solved')
        self.assertTrue(r['state_committed'])
        self.assertEqual(r['scene']['state'],r['plan']['final_state'])
        self.assertTrue(r['scene']['state']['safe_route_identified'])
        self.assertFalse(r['scene']['state']['gate_open'])
        self.assertEqual(r['scene']['ai_detail_count'],0)
        self.assertGreater(r['scene']['character_mentions']['Arin'],0)
        self.assertGreater(r['scene']['character_mentions']['Elara'],0)
        self.assertNotIn('opened the northern gate', r['scene']['text'])
        self.assertEqual([s['action_id'] for s in r['plan']['plan']], ['scout_wall','find_side_entrance','mark_safe_route'])

    def test_plan_is_verified_before_composition(self):
        spec=copy.deepcopy(self.spec)
        spec['goal']={'father_found':True}
        r=compose(spec,self.bridge)
        self.assertEqual(r['status'],'unsatisfiable')
        self.assertFalse(r['state_committed'])
        self.assertIsNone(r['scene'])

    def test_forbidden_cheaper_action_not_accepted(self):
        r=compose(self.spec,self.bridge)
        self.assertNotIn('force_northern_gate',[s['action_id'] for s in r['plan']['plan']])

    def test_name_mismatch_rejected(self):
        bridge=copy.deepcopy(self.bridge)
        bridge['characters']['arin']['name']='Caden'
        with self.assertRaises(IntegrationError): compose(self.spec,bridge)

    def test_action_mapping_missing_rejected(self):
        bridge=copy.deepcopy(self.bridge)
        del bridge['action_beats']['find_side_entrance']
        with self.assertRaises(IntegrationError): compose(self.spec,bridge)

    def test_beat_cannot_override_effects(self):
        bridge=copy.deepcopy(self.bridge)
        bridge['action_beats']['scout_wall']['set_state']={'gate_open':True}
        with self.assertRaises(IntegrationError): compose(self.spec,bridge)

    def test_prior_state_accepted_and_goal_already_reached(self):
        first=compose(self.spec,self.bridge)
        second=compose(self.spec,self.bridge,previous_state=first['scene']['state'])
        self.assertEqual(second['status'],'solved_no_actions')
        self.assertFalse(second['state_committed'])

    def test_reject_unknown_previous_facts(self):
        with self.assertRaises(IntegrationError): compose(self.spec,self.bridge,previous_state={'unknown':True})

    def test_unsupported_goal_state_wont_fake_story(self):
        spec=copy.deepcopy(self.spec)
        spec['initial_state']['gate_open']=True
        with self.assertRaises(Exception): compose(spec,self.bridge)

    def test_reject_inconsistent_actor(self):
        bridge=copy.deepcopy(self.bridge)
        bridge['action_beats']['scout_wall']['actor']='arin'
        with self.assertRaises(IntegrationError): compose(self.spec,bridge)

if __name__=='__main__': unittest.main()
