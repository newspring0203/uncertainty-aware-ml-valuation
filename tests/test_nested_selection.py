import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_nested_ridge import model_for,select_candidate
from run_ridge_robustness import clip_numeric
class NestedSelectionChecks(unittest.TestCase):
 def test_clip_preserves_input_and_cohort_features(self):
  x=np.array([[20.,-20.,0.,1.,2.,3.,4.,1.,0.]])
  out=clip_numeric(x)
  self.assertEqual(x[0,0],20.)
  self.assertEqual(out[0,0],5.);self.assertEqual(out[0,1],-5.)
  np.testing.assert_array_equal(out[:,7:],x[:,7:])
 def test_candidate_configuration(self):
  self.assertEqual(model_for('alpha100').steps[-1][1].alpha,100.)
  self.assertEqual(len(model_for('clip5_alpha0.1').steps),4)
 def test_inner_scores_cover_each_training_row(self):
  rows=[dict(firm_id=str(i).zfill(8),cohort='A' if i<6 else 'B',X=[float(i),.1,.5,.2,.3,.1,.2,float(i<6),float(i>=6)],target=1+i/10) for i in range(12)]
  chosen,scores=select_candidate(rows,0,0)
  self.assertIn(chosen,{r['candidate'] for r in scores})
  self.assertTrue(all(r['inner_n']==12 for r in scores))
if __name__=='__main__':unittest.main()
