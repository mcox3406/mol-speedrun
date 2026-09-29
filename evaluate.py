"""Frozen v0 scoring. Predictions: id,S1_eV,T1_eV,log_f (log10(1+f/0.001))."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
PROTOCOL=json.loads((ROOT/'protocol.json').read_text())
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_rows(path):
 with Path(path).open(newline='') as f:rows=list(csv.DictReader(f))
 ids=[r['id'] for r in rows]
 if not rows or len(set(ids))!=len(ids):raise ValueError('Empty file or duplicate IDs')
 return rows

def score_arrays(pred,truth):
 pred=np.asarray(pred,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
 if pred.shape!=truth.shape or pred.ndim!=2 or pred.shape[1]!=3 or not len(pred):raise ValueError('Expected matching nonempty N x 3 arrays')
 if not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite prediction or label')
 if (truth[:,:2]<=0).any() or (truth[:,2]<0).any():raise ValueError('Nonphysical target')
 true_log=np.log10(1+truth[:,2]/PROTOCOL['intensity_floor']);raw=np.maximum(0,PROTOCOL['intensity_floor']*(10**np.clip(pred[:,2],-6,6)-1));bright=truth[:,2]>=PROTOCOL['bright_threshold']
 if not bright.any():raise ValueError('No strong transitions to score')
 return {'S1_eV':float(np.abs(pred[:,0]-truth[:,0]).mean()),'T1_eV':float(np.abs(pred[:,1]-truth[:,1]).mean()),'delta_ST_eV':float(np.abs((pred[:,0]-pred[:,1])-(truth[:,0]-truth[:,1])).mean()),'log_f':float(np.abs(pred[:,2]-true_log).mean()),'raw_f':float(np.abs(raw-truth[:,2]).mean()),'bright_f':float(np.abs(raw[bright]-truth[bright,2]).mean()),'bright_n':int(bright.sum())}
def qualifies(scores):return all(np.isfinite(scores[k]) and scores[k]<=v for k,v in PROTOCOL['targets'].items())
def score_files(labels,predictions):
 truth=read_rows(labels);pred=read_rows(predictions)
 if [r['id'] for r in truth]!=[r['id'] for r in pred]:raise ValueError('Prediction IDs must match all label IDs in order')
 return score_arrays([[float(r[k]) for k in ['S1_eV','T1_eV','log_f']] for r in pred],[[float(r[k]) for k in ['S1_eV','T1_eV','f_S1']] for r in truth])
def main():
 p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);a=p.parse_args();s=score_files(a.labels,a.predictions);print(json.dumps({'metrics':s,'passes_validation_targets':qualifies(s)},indent=2,allow_nan=False))
if __name__=='__main__':main()
