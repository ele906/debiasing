import unittest
import numpy as np
from common import prompt_geometry
from infer import iou

class PilotTests(unittest.TestCase):
 def landmarks(self):
  p=np.zeros((468,3));p[234,:2]=(.3,.4);p[454,:2]=(.7,.4);p[152,:2]=(.5,.7);p[10,:2]=(.5,.2);p[17,:2]=(.5,.55);p[13,:2]=(.5,.53);p[14,:2]=(.5,.54);return p
 def test_one_policy_uses_five_positive_three_negative(self):
  c=prompt_geometry(self.landmarks(),512,512)
  self.assertEqual(c['labels'],[1]*5+[0]*3);self.assertEqual(len(c['points']),8)
  self.assertLess(c['box'][0],c['box'][2]);self.assertLess(c['box'][1],c['box'][3])
 def test_scale_equivariance_and_no_group_input(self):
  a=prompt_geometry(self.landmarks(),512,512);b=prompt_geometry(self.landmarks(),1024,1024)
  np.testing.assert_allclose(np.asarray(a['points'])*2,b['points']);np.testing.assert_allclose(np.asarray(a['box'])*2,b['box'])
 def test_iou_edge_cases(self):
  a=np.zeros((3,3),bool);b=a.copy();b[0,0]=True
  self.assertEqual(iou(a,a),1.);self.assertEqual(iou(a,b),0.);self.assertEqual(iou(b,b),1.)
if __name__=='__main__':unittest.main()
