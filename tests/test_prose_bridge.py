import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from training.v16.plan_to_scene import compose
from training.v16.registry_writer import validate_detail

ROOT = Path(__file__).resolve().parents[1]

class StubModel:
    def __init__(self, candidate, raw=None):
        self.candidate = candidate
        self.raw = raw or candidate
        self.last_generation = {}
    def generate(self, location, previous, max_tokens):
        self.last_generation = {'prompt_style':'continuation', 'prompt':'test input', 'raw_output':self.raw}
        return self.candidate

class TestProseBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario=json.loads((ROOT/'examples/castle_reasoning.json').read_text(encoding='utf8'))
        cls.bridge=json.loads((ROOT/'examples/castle_plan_to_scene.json').read_text(encoding='utf8'))

    def test_base_planner_unchanged(self):
        result=compose(self.scenario,self.bridge)
        self.assertEqual(result['status'],'solved')
        self.assertEqual(result['scene']['ai_detail_count'],0)
        self.assertFalse(result['scene']['state']['gate_open'])

    def test_rejected_output_still_in_report(self):
        model=StubModel('M.', 'M. And more unrelated prose.')
        result=compose(self.scenario,self.bridge,model=model)
        infos=[r['detail'] for r in result['scene']['trace'] if r['detail']['status']=='rejected']
        self.assertTrue(infos)
        self.assertEqual(infos[0]['model_debug']['raw_output'],'M. And more unrelated prose.')
        self.assertNotIn('M.',result['scene']['text'])
        self.assertEqual(result['scene']['ai_detail_count'],0)

    def test_safe_detail_accepted_and_provenance_preserved(self):
        model=StubModel('the rain rattled against the ancient stone.', 'the rain rattled against the ancient stone.')
        result=compose(self.scenario,self.bridge,model=model)
        self.assertGreater(result['scene']['ai_detail_count'],0)
        self.assertIn('the rain rattled', result['scene']['text'])
        self.assertFalse(result['scene']['generated_full_scene_by_model'])

    def test_unrelated_names_rejected(self):
        story={'characters':{'arin':{'name':'Arin'},'elara':{'name':'Elara'}},'forbidden_phrases':[]}
        self.assertTrue(validate_detail('Evan opened a door beside the castle.',story,[]))
        self.assertTrue(validate_detail('...',story,[]))

if __name__=='__main__': unittest.main()
