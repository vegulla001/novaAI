import unittest
from scripts.build_v15_instruction_data import make_rows,format_prompt
class TestData(unittest.TestCase):
 def test_pairs_disjoint(self):
  a,b=make_rows(73,100,50)
  x={(r['name'],r['place']) for r in a}
  y={(r['name'],r['place']) for r in b}
  self.assertFalse(x&y)
 def test_variety_and_determinism(self):
  a,b=make_rows(73,100,50)
  c,d=make_rows(73,100,50)
  self.assertEqual((a,b),(c,d))
  self.assertGreater(len({r['response'] for r in a}),80)
 def test_prompt(self):
  self.assertTrue(format_prompt('Aren','finds','map','forest').endswith('Response: '))
if __name__=='__main__':unittest.main()
