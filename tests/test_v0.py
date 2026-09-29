import csv,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import evaluate as ev
import validate_submission as vs

class ScoringTests(unittest.TestCase):
 def test_exact_predictions_and_brightness_boundary(self):
  y=np.array([[3,2,.1],[4,2,.01]]);p=y.copy();p[:,2]=np.log10(1+y[:,2]/.001);s=ev.score_arrays(p,y)
  self.assertEqual(s['bright_n'],1);self.assertTrue(ev.qualifies(s));self.assertLess(s['raw_f'],1e-15)
 def test_gap_is_derived(self):
  y=np.array([[3,2,.1]]);p=np.array([[3.2,1.9,np.log10(101)]]);self.assertAlmostEqual(ev.score_arrays(p,y)['delta_ST_eV'],.3)
 def test_reject_nonfinite_and_shape(self):
  for p in [np.array([[np.nan,2,1]]),np.zeros((2,3))]:
   with self.assertRaises(ValueError):ev.score_arrays(p,np.array([[3,2,.1]]))
 def test_every_gate_required(self):
  s=dict(ev.PROTOCOL['targets']);self.assertTrue(ev.qualifies(s));s['bright_f']+=1e-5;self.assertFalse(ev.qualifies(s))

class SubmissionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.paths=[];self.patch=patch.object(vs,'ROOT',self.root);self.patch.start()
  (self.root/'protocol.json').write_bytes((ev.ROOT/'protocol.json').read_bytes());rows=[['a',3,2,.1],['b',4,2,.01]]
  for part in ['validation','audit']:
   with (self.root/f'{part}.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['id','S1_eV','T1_eV','f_S1']);w.writerows(rows)
  (self.root/'data-manifest.json').write_text(json.dumps({'files':{f'{p}.csv':{'sha256':ev.digest(self.root/f'{p}.csv')} for p in ['validation','audit']}}))
  for seed in ev.PROTOCOL['seeds']:
   pred=self.root/f'{seed}.csv'
   with pred.open('w',newline='') as f:w=csv.writer(f);w.writerow(['id','S1_eV','T1_eV','log_f']);w.writerows([r[0],r[1],r[2],math.log10(1+r[3]/.001)] for r in rows)
   scores=ev.score_files(self.root/'validation.csv',pred);r=dict(protocol=ev.PROTOCOL['id'],protocol_sha256=ev.digest(self.root/'protocol.json'),manifest_sha256=ev.digest(self.root/'data-manifest.json'),seed=seed,status='qualified',gpu='NVIDIA L40S',gpu_count=1,cpu_threads=4,total_seconds=10,preprocessing_seconds=1,history=[dict(seconds=9,metrics=scores)],predictions_file=pred.name,predictions_sha256=ev.digest(pred),metrics=scores,commit='a'*40,source_sha256={'trainer.py':'b'*64},checkpoint_file=f'{seed}.pt',checkpoint_sha256='c'*64);path=self.root/f'{seed}.json';path.write_text(json.dumps(r));self.paths.append(path)
  self.audit=self.root/'audit.json';(self.root/'audit.predictions.csv').write_bytes((self.root/f'{ev.PROTOCOL["seeds"][0]}.csv').read_bytes());self.audit.write_text(json.dumps(dict(protocol=ev.PROTOCOL['id'],seed=ev.PROTOCOL['seeds'][0],run_sha256=ev.digest(self.paths[0]),checkpoint_sha256='c'*64,predictions_sha256=ev.digest(self.root/'audit.predictions.csv'),metrics=scores)))
 def tearDown(self):self.patch.stop();self.temp.cleanup()
 def check(self):return vs.validate_reports(self.paths,self.root,self.audit)
 def test_valid_panel_is_unverified(self):self.assertEqual(self.check()['status'],'checks_passed_pending_reproduction')
 def test_duplicate_seed_rejected(self):
  self.paths[2]=self.paths[1]
  with self.assertRaises(ValueError):self.check()
 def test_wrong_reported_score_rejected(self):
  p=self.paths[0];r=json.loads(p.read_text());r['metrics']['S1_eV']=.1;p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):self.check()
 def test_data_tampering_rejected(self):
  (self.root/'validation.csv').write_text('bad')
  with self.assertRaises(ValueError):self.check()
 def test_wrong_audit_checkpoint_rejected(self):
  r=json.loads(self.audit.read_text());r['checkpoint_sha256']='d'*64;self.audit.write_text(json.dumps(r))
  with self.assertRaises(ValueError):self.check()
 def test_artifact_path_traversal_rejected(self):
  with self.assertRaises(ValueError):vs.local_artifact(self.paths[0],'../secret')
 def test_out_of_budget_rejected(self):
  p=self.paths[0];r=json.loads(p.read_text());r['total_seconds']=3601;p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):self.check()
 def test_changed_recipe_rejected(self):
  p=self.paths[1];r=json.loads(p.read_text());r['source_sha256']['trainer.py']='e'*64;p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):self.check()
if __name__=='__main__':unittest.main()
