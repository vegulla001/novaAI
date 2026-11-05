import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
p=Path(__file__).resolve().parents[1]/'scripts'/'build_scene_sft_v15.py'
spec=importlib.util.spec_from_file_location('scene_builder',p)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class FakeTokenizer:
    def token_to_id(self,s):return {'<bos>':1,'<eos>':2}[s]
    def encode(self,text,add_special_tokens=False):
        class E:pass
        e=E();e.ids=[10+ord(x)%128 for x in text];return e

class TestSceneSFT(unittest.TestCase):
    def row(self):
        return {'story_id':'story-A','reviewed':True,
                'spec':{'title':'Example','characters':[{'name':'Aren','role':'warrior'}]},
                'chapter':{'number':1,'title':'The Map'},
                'scene':{'number':1,'beat':'Aren finds a map','goal':'Find it','obstacle':'Rain','outcome':'He obtains it'},
                'accepted_events':[], 'response':'Aren saw a map beneath the stones. Rain fell on the road.'}
    def test_unreviewed_rejected(self):
        r=self.row();r['reviewed']=False
        self.assertIn('not_reviewed',mod.quality(r,1))
    def test_outline_rejected(self):
        r=self.row();r['response']='1. Goal: Aren finds a map.'
        self.assertIn('outline_instead_of_prose',mod.quality(r,1))
    def test_mask_and_shift(self):
        r=self.row();encoded,details=mod.assemble(r,FakeTokenizer(),10000)
        ids,labels=encoded['input_ids'],encoded['labels']
        prompt_end=1+details['prompt_tokens']
        self.assertTrue(all(v==-100 for v in labels[:prompt_end]))
        self.assertEqual(ids[prompt_end:],labels[prompt_end:])
        self.assertEqual(ids[-1],2)
    def test_overlong_refused(self):
        with self.assertRaisesRegex(ValueError,'exceeds_context_window'):
            mod.assemble(self.row(),FakeTokenizer(),100)
    def test_split_stable(self):
        self.assertEqual(mod.split_of('story-A',20),mod.split_of('story-A',20))
if __name__=='__main__':unittest.main()
