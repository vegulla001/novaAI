import unittest
from scripts.v15_name_anchor import anchor_name, extract_requested_name

class NameAnchorTests(unittest.TestCase):
    def test_unseen_name(self):
        prompt='Instruction: Write exactly one sentence about Keziah who finds an amulet at the bridge.\nResponse: '
        response='Kiran found an amulet at the bridge.'
        fixed, changed, _ = anchor_name(prompt, response)
        self.assertTrue(changed)
        self.assertEqual(fixed, 'Keziah found an amulet at the bridge.')

    def test_other_slots_remain_wrong(self):
        prompt='Instruction: Write exactly one sentence about Saanvi who guards a key at the palace.\nResponse: '
        fixed, changed, _ = anchor_name(prompt, 'Sera examined a compass at the room.')
        self.assertTrue(changed)
        self.assertEqual(fixed, 'Saanvi examined a compass at the room.')

    def test_no_repair_of_nonsense(self):
        prompt='Instruction: Write exactly one sentence about Keziah who finds a map at the castle.\nResponse: '
        fixed, changed, _ = anchor_name(prompt, 'The champion of the universe')
        self.assertFalse(changed)
        self.assertEqual(fixed, 'The champion of the universe')

    def test_paraphrases(self):
        prompt='Instruction: Tell me in one past-tense sentence what happened when Odessa opens a map at the tower.\nResponse: '
        self.assertEqual(extract_requested_name(prompt), 'Odessa')
        prompt='Instruction: Summarize in a single sentence: At the valley, Ryker finds a ring.\nResponse: '
        self.assertEqual(extract_requested_name(prompt), 'Ryker')

    def test_unknown_prompt(self):
        self.assertEqual(anchor_name('Write a poem', 'Sera found a key at the tower.')[1], False)

if __name__ == '__main__': unittest.main()
