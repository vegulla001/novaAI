import unittest
import torch
from training.v16.diagnose_context import check_sft_rows, token_nll

class ToyLM(torch.nn.Module):
    def forward(self,x):
        b,t=x.shape
        logits=torch.zeros(b,t,20)
        for k in range(20):
            logits[:,:,k] = (x==k).float()*4
        return logits,None

class Tests(unittest.TestCase):
    def test_alignment(self):
        r=[{'input_ids':[2,3,4,5], 'labels':[-100,-100,4,5]}]
        out=check_sft_rows(r)
        self.assertEqual(out['counts']['average_prompt_tokens_before_crop'],2)
        self.assertTrue(out['examples'][0]['first_target_equals_input'])
    def test_bad_lengths(self):
        out=check_sft_rows([{'input_ids':[1,2], 'labels':[1]}]);self.assertEqual(out['counts']['bad_lengths'],1)
    def test_context_nll(self):
        m=ToyLM()
        a=token_nll(m,[2,3],[3],8,'cpu')
        b=token_nll(m,[2,4],[3],8,'cpu')
        self.assertLess(a,b)
    def test_context_limit(self):
        with self.assertRaises(ValueError): token_nll(ToyLM(),[1,2,3],[4,5],4,'cpu')
if __name__=='__main__':unittest.main()

class LastOnlyLM(ToyLM):
    def forward(self, x):
        logits, _ = super().forward(x)
        return logits[:, -1:, :], None

class ForwardContractTests(unittest.TestCase):
    def test_last_only_same_nll_as_full_sequence(self):
        prompt, completion = [2, 3, 4], [4, 5, 6, 7]
        full = token_nll(ToyLM(), prompt, completion, 16, 'cpu')
        last = token_nll(LastOnlyLM(), prompt, completion, 16, 'cpu')
        self.assertAlmostEqual(full, last, places=5)

    def test_last_only_prompt_affects_score(self):
        good = token_nll(LastOnlyLM(), [2, 3], [3], 16, 'cpu')
        wrong = token_nll(LastOnlyLM(), [2, 4], [3], 16, 'cpu')
        self.assertLess(good, wrong)

    def test_last_only_multiple_completion_tokens(self):
        good = token_nll(LastOnlyLM(), [2, 3], [3, 3, 3], 16, 'cpu')
        wrong = token_nll(LastOnlyLM(), [2, 4], [3, 3, 3], 16, 'cpu')
        self.assertLess(good, wrong)
