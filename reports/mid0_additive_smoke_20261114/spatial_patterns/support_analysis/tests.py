"""Analysis math tests; synthetic arrays only, no models or image generation."""
import unittest
import numpy as np
import analysis as a

class AnalysisTests(unittest.TestCase):
    def test_exact_bootstrap_enumeration(self):
        samples=list(a.seed_bootstrap_counts(list(range(5))))
        self.assertEqual(len(samples),126)
        self.assertEqual(sum(m for _,m in samples),3125)
        self.assertTrue(all(sum(c.values())==5 for c,_ in samples))
    def test_controlled_predictor_is_unidentified(self):
        D=np.column_stack((np.ones(6),[0,0,0,1,1,1]))
        x=2+D[:,1];y=(3*x)[:,None]
        result=a.residual_fit(D,x,y,np.ones(6))
        self.assertIsNone(result['beta'])
    def test_fwl_recovers_known_conditional_slope(self):
        D=np.column_stack((np.ones(6),[0,0,0,1,1,1]))
        x=np.array([0,1,2,2,3,4.]);y=(3*x+5*D[:,1])[:,None]
        result=a.residual_fit(D,x,y,np.ones(6))
        self.assertAlmostEqual(result['beta'][0],3.)
    def test_full_sparse_floor_does_not_bound_selected_error(self):
        latent=np.array([2.]+[1.]*10)
        sparse=latent.copy();sparse[1]=0
        self.assertEqual(np.count_nonzero(latent),11)
        self.assertEqual(np.count_nonzero(sparse),10)
        self.assertEqual(abs(sparse[0]-latent[0]),0)
        self.assertEqual(np.sqrt(np.partition(latent**2,-10)[:-10].sum()),1.)
    def test_only_one_coordinate_support_identity(self):
        old=np.array([[2,1,0,0],[0,1,2,0],[2,1,0,0],[0,1,2,0]],float)
        p=np.array([1,1,0,1.]);new=old.copy();new[:,0]+=p
        expansion=np.mean(np.count_nonzero(new,axis=1)>2)
        formula=np.mean(p!=0)-np.mean((p!=0)&(old[:,0]!=0))
        self.assertEqual(expansion,formula)
    def test_actual_dataset_balance(self):
        import csv
        from pathlib import Path
        with (Path(__file__).parent/'results/per_image_analysis.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),200)
        self.assertEqual(len({r['seed'] for r in rows}),5)
        for method in a.METHODS:
            self.assertAlmostEqual(sum(float(r['balance_weight']) for r in rows if r['method']==method),20.)

if __name__=='__main__':unittest.main()
