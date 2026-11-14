import unittest
from analyze import interval

class StatisticsTests(unittest.TestCase):
    def test_five_seed_exact_two_sided_resolution(self):
        mean,lo,hi,p=interval([1,1,1,1,1])
        self.assertEqual((mean,lo,hi),(1.,1.,1.))
        self.assertEqual(p,2/32)
    def test_symmetric_zero_effect(self):
        mean,lo,hi,p=interval([-2,-1,0,1,2])
        self.assertEqual(mean,0.)
        self.assertLess(lo,0);self.assertGreater(hi,0)
        self.assertEqual(p,1.)
    def test_missing_values_not_invented(self):
        self.assertEqual(interval([]),(None,None,None,None))
if __name__=='__main__':unittest.main()
