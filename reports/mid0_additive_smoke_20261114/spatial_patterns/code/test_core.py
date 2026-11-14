import unittest
import numpy as np
import torch
from core import apply_pattern,match_pattern

class PatternTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.original=torch.randn(1,8,16,16,dtype=torch.float16)
        self.z=torch.zeros(1,16,16,1);self.direction=torch.randn(8,1)
    def test_zero_noop(self):
        _,edited,_=apply_pattern(self.original,self.z,torch.ones_like(self.z),0,self.direction)
        self.assertTrue(torch.equal(edited,self.original))
    def test_sparse_and_uniform_match_actual_norm(self):
        sparse=torch.zeros_like(self.z);sparse[:,5:8,6:9]=2
        for p in (sparse,torch.ones_like(self.z)):
            coefficient,error,passed=match_pattern(self.original,self.z,p,42.5242,self.direction)
            self.assertTrue(passed)
            modified,edited,_=apply_pattern(self.original,self.z,p,coefficient,self.direction)
            self.assertLess(abs(float((edited.float()-self.original.float()).norm())-42.5242)/42.5242,.005)
            self.assertTrue(torch.equal(modified,coefficient*p))
    def test_shuffle_preserves_exact_values(self):
        p=np.arange(256,dtype=np.float32);shuffled=p[np.random.default_rng(1729).permutation(256)]
        self.assertTrue(np.array_equal(np.sort(p),np.sort(shuffled)))
        self.assertFalse(np.array_equal(p,shuffled))
    def test_reject_zero_pattern(self):
        with self.assertRaises(ValueError):match_pattern(self.original,self.z,torch.zeros_like(self.z),42,self.direction)
if __name__=='__main__':unittest.main()
