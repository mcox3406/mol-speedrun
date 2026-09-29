import unittest
import torch
from calibrate import Regressor,four
class CalibrationTests(unittest.TestCase):
 def test_padding_does_not_change_prediction(self):
  torch.manual_seed(0);model=Regressor(width=32,layers=2,heads=4,dropout=0).eval();tokens=torch.tensor([[68,69,70],[68,70,0]])
  with torch.no_grad():
   a=model(tokens);b=model(torch.nn.functional.pad(tokens,(0,4)))
  torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
 def test_gap_is_difference_and_loss_backpropagates(self):
  torch.manual_seed(0);model=Regressor(width=32,layers=1,heads=4);p=model(torch.tensor([[68,69,70],[68,70,0]]));targets=four(p)
  torch.testing.assert_close(targets[:,2],p[:,0]-p[:,1]);targets.square().mean().backward()
  self.assertTrue(torch.isfinite(model.token.weight.grad).all());self.assertGreater(model.token.weight.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
