import unittest
import torch
from additive import apply_additive,match_offset,apply_set

class AdditiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.original=torch.randn(1,8,4,4,dtype=torch.float16)
        self.z=torch.zeros(1,4,4,1)
        self.direction=torch.randn(8,1)
    def test_inactive_cells_activate_without_topk(self):
        changed,edited,delta=apply_additive(self.original,self.z,0.2,self.direction)
        self.assertTrue(torch.all(changed==0.2))
        self.assertGreater(float((edited-self.original).norm()),0)
    def test_signed_offsets_are_not_clipped(self):
        changed,_,_=apply_additive(self.original,self.z,-0.2,self.direction)
        self.assertTrue(torch.all(changed<0))
    def test_zero_is_exact_noop(self):
        _,edited,_=apply_additive(self.original,self.z,0.,self.direction)
        self.assertTrue(torch.equal(edited,self.original))
    def test_set_replaces_active_and_inactive_cells_exactly(self):
        z=self.z.clone();z[:,1:3]=17.
        for target in (-200.,-10.,0.,10.,200.):
            changed,edited,delta=apply_set(self.original,z,target,self.direction)
            self.assertTrue(torch.all(changed==target))
            expected=(torch.full_like(z,target)-z)@self.direction.T
            self.assertTrue(torch.equal(delta,expected))

    def test_actual_norm_matches_across_contexts(self):
        for original in (self.original,self.original*5):
            beta,error=match_offset(original,self.z,2.,self.direction)
            _,edited,_=apply_additive(original,self.z,beta,self.direction)
            self.assertLess(error,.005)
            self.assertAlmostEqual(float((edited.float()-original.float()).norm()),2.,delta=.01)

if __name__=='__main__':unittest.main()
