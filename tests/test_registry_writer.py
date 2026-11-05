import copy
import json
import unittest
from pathlib import Path
from training.v16.registry_writer import StoryError, generate_story, validate_detail

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/scene_castle_registry.json'
class TestWriter(unittest.TestCase):
    def setUp(self):
        self.story = json.loads(EXAMPLE.read_text(encoding='utf8'))

    def test_no_gpu_and_all_cast(self):
        result = generate_story(self.story)
        self.assertGreaterEqual(result['character_mentions']['Arin'], 3)
        self.assertGreaterEqual(result['character_mentions']['Elara'], 3)
        self.assertEqual(result['ai_detail_count'], 0)
        self.assertFalse(result['state']['northern_gate_open'])
        self.assertEqual(result['state']['route'], 'outer_wall')
        self.assertFalse(result['generated_full_scene_by_model'])

    def test_unknown_characters_fail(self):
        s = copy.deepcopy(self.story)
        s['beats'][0]['actor'] = 'evan'
        with self.assertRaises(StoryError):
            generate_story(s)

    def test_missing_required_character_fail(self):
        s = copy.deepcopy(self.story)
        s['beats'] = [s['beats'][0]]
        with self.assertRaises(StoryError):
            generate_story(s)

    def test_model_detail_rejection(self):
        class FakeModel:
            def generate(self, *args): return 'Evan opened the northern gate.'
        result = generate_story(self.story, FakeModel())
        self.assertEqual(result['ai_detail_count'], 0)
        self.assertEqual(result['trace'][0]['detail']['status'], 'rejected')

    def test_model_detail_acceptance(self):
        class FakeModel:
            i=0
            def generate(self, *args):
                self.i += 1
                return ['Rain tapped on the stones.', 'Wind moved across the walls.', 'Water dripped from the arch.'][self.i-1]
        result = generate_story(self.story, FakeModel())
        self.assertEqual(result['ai_detail_count'], 3)
        self.assertGreaterEqual(result['character_mentions']['Arin'], 3)

    def test_fragment_details_rejected(self):
        for bad in ('M.', '...', 'Wind.', 'The air was cold.', 'Rain...'): 
            self.assertTrue(validate_detail(bad, self.story, []), bad)

    def test_fragment_not_inserted(self):
        class FakeModel:
            def generate(self, *args): return 'M.'
        result = generate_story(self.story, FakeModel())
        self.assertEqual(result['ai_detail_count'], 0)
        self.assertNotIn('\n\nM.\n', result['text'])

    def test_detail_cannot_introduce_names(self):
        self.assertTrue(validate_detail('Evan watched from the wall.', self.story, []))
        self.assertTrue(validate_detail('Arin moved the lantern.', self.story, []))
        self.assertEqual(validate_detail('Rain tapped on the stones.', self.story, []), [])

if __name__ == '__main__': unittest.main()
