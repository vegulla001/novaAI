import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'training' / 'v16' / 'cpu_reasoning_generalization_saved.py'
spec = importlib.util.spec_from_file_location('visible_reasoning', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class VisibleReasoningTests(unittest.TestCase):
    def test_choose_complete_counterfactual_pair(self):
        cases = [
            dict(skill='gate', pair_id='g1', answer='yes', question='Open?', choices=['yes','no']),
            dict(skill='gate', pair_id='g1', answer='no', question='Open?', choices=['yes','no']),
        ]
        self.assertEqual(set(mod.choose_pairs(cases, 1, 1)), {0,1})

    def test_invalid_pair_is_rejected(self):
        cases = [
            dict(skill='gate', pair_id='g1', answer='yes', question='Open?', choices=['yes','no']),
            dict(skill='gate', pair_id='g1', answer='yes', question='Open?', choices=['yes','no']),
        ]
        with self.assertRaises(ValueError):
            mod.choose_pairs(cases, 1, 1)

    def test_shifted_supervision_mask(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        x, y = mod.make_batch([{'input_ids':[10, 11, 12, 13], 'labels':[-100,-100,12,13]}], 'cpu')
        self.assertEqual(x.tolist(), [[10,11,12]])
        self.assertEqual(y.tolist(), [[-1,12,13]])

if __name__=='__main__':
    unittest.main()
