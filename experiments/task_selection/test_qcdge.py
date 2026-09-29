"""Scientific parsing checks for interleaved spin states and explicit units."""
import json,unittest
from extract_qcdge import parse_states
class ExcitationParsing(unittest.TestCase):
 def states(self):
  # Lowest overall state is triplet; dictionary order is deliberately reversed.
  rows={}
  for i in range(10):
   rows[str(2*i+1)]=dict(state=2*i+1,state_type='Triplet',oscillator_trength=0,excitation_e_eV=f'{1+i} eV')
   rows[str(2*i+2)]=dict(state=2*i+2,state_type='Singlet',oscillator_trength=.12,excitation_e_eV=f'{3+i} eV')
  return dict(reversed(list(rows.items())))
 def test_spin_resolved_minima(self):
  self.assertEqual(parse_states(json.dumps(self.states())),(3.,1.,2.,.12))
 def test_wrong_units_rejected(self):
  d=self.states();d['1']['excitation_e_eV']='1 Hartree'
  with self.assertRaises(ValueError):parse_states(json.dumps(d))
 def test_missing_spin_state_rejected(self):
  d=self.states();del d['1']
  with self.assertRaises(ValueError):parse_states(json.dumps(d))
 def test_nonfinite_rejected(self):
  d=self.states();d['1']['excitation_e_eV']='nan eV'
  with self.assertRaises(ValueError):parse_states(json.dumps(d))
if __name__=='__main__':unittest.main()
