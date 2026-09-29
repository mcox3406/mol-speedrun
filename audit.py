"""Evaluate a frozen reference checkpoint on rare families, outside the race clock."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from model import Regressor
from evaluate import ROOT,digest,read_rows,score_arrays

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/qcdge-v0'));p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4);r=json.loads(a.run.read_text());checkpoint=a.run.parent/r['checkpoint_file']
 if digest(checkpoint)!=r['checkpoint_sha256']:raise ValueError('Checkpoint does not match frozen run')
 path=a.data/'audit.csv';manifest=json.loads((ROOT/'data-manifest.json').read_text())
 if digest(path)!=manifest['files']['audit.csv']['sha256']:raise ValueError('Audit data checksum mismatch')
 rows=read_rows(path);state=torch.load(checkpoint,map_location=a.device,weights_only=True);model=Regressor().to(a.device);model.load_state_dict(state['model']);model.eval();pred=[]
 with torch.no_grad():
  for start in range(0,len(rows),256):
   rr=rows[start:start+256];codes=[[ord(c)+1 for c in x['smiles']] for x in rr];x=torch.zeros(len(rr),max(map(len,codes)),dtype=torch.long,device=a.device)
   for i,v in enumerate(codes):x[i,:len(v)]=torch.tensor(v,device=a.device)
   with torch.autocast(device_type=a.device,dtype=torch.bfloat16,enabled=a.device=='cuda'):out=model(x)
   pred.append((out.float()*state['std']+state['mean']).cpu().numpy())
 pred=np.concatenate(pred);truth=np.asarray([[float(r[k]) for k in ['S1_eV','T1_eV','f_S1']] for r in rows]);scores=score_arrays(pred,truth);out=a.output.with_suffix('.predictions.csv');a.output.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['id','S1_eV','T1_eV','log_f']);w.writerows([r['id'],*map(float,v)] for r,v in zip(rows,pred))
 a.output.write_text(json.dumps(dict(protocol=r['protocol'],seed=r['seed'],run_sha256=digest(a.run),checkpoint_sha256=digest(checkpoint),predictions_sha256=digest(out),metrics=scores,rows=len(rows),scope='rare-family audit only; not used for selection or speed ranking'),indent=2)+'\n');print(scores)
if __name__=='__main__':main()
