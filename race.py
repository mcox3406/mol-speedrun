"""Train from scratch to every v0 target. The clock includes data processing and exports."""
import argparse,csv,json,os,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import Regressor,four
from evaluate import PROTOCOL,ROOT,digest,read_rows,score_arrays,qualifies

def revision():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
 except (OSError,subprocess.CalledProcessError):return os.environ.get('SOURCE_COMMIT','unknown')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/qcdge-v0'));p.add_argument('--seed',type=int,choices=PROTOCOL['seeds'],default=PROTOCOL['seeds'][0]);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(4);cuda=a.device=='cuda'
 if not cuda and not a.smoke:raise ValueError('Official reference runs require CUDA; use --smoke for CPU checks')
 if cuda:torch.cuda.init();torch.cuda.synchronize();torch.set_float32_matmul_precision('high')
 def sync():
  if cuda:torch.cuda.synchronize()
 commit=revision();started=time.perf_counter();manifest=json.loads((a.data/'manifest.json').read_text());frozen=json.loads((ROOT/'data-manifest.json').read_text())
 if manifest!=frozen:raise ValueError('Dataset manifest differs from frozen v0')
 rows={}
 for part in ['train','validation']:
  path=a.data/f'{part}.csv'
  if digest(path)!=manifest['files'][path.name]['sha256']:raise ValueError('Dataset checksum mismatch')
  rows[part]=read_rows(path)
  if len(rows[part])!=manifest['files'][path.name]['rows']:raise ValueError('Dataset row count mismatch')
  if a.smoke:rows[part]=rows[part][:256]
 def encode(rr):
  codes=[[ord(c)+1 for c in r['smiles']] for r in rr];length=max(map(len,codes))
  if length>512 or any(not v or max(v)>128 for v in codes):raise ValueError('Unsupported SMILES encoding')
  x=torch.zeros(len(rr),length,dtype=torch.long)
  for i,v in enumerate(codes):x[i,:len(v)]=torch.tensor(v)
  return x.to(a.device)
 X={k:encode(v) for k,v in rows.items()};truth={k:np.array([[float(r[t]) for t in ['S1_eV','T1_eV','f_S1']] for r in rr],dtype=np.float64) for k,rr in rows.items()};y=torch.tensor(truth['train'],dtype=torch.float32,device=a.device);y[:,2]=torch.log10(1+y[:,2]/.001);Y=four(y);mu=y.mean(0);sd=y.std(0,unbiased=False);scale=Y.std(0,unbiased=False);torch.manual_seed(a.seed);model=Regressor().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01,fused=cuda);sync();prep=time.perf_counter()-started;history=[];passed=False;budget=PROTOCOL['limits']['seconds'];batch_size=256
 for epoch in range(1,2 if a.smoke else 10001):
  model.train();order=torch.randperm(len(y),device=a.device);complete=True
  for batch in order.split(batch_size):
   with torch.autocast(device_type=a.device,dtype=torch.bfloat16,enabled=cuda):pred=model(X['train'][batch])
   loss=((four(pred.float()*sd+mu)-Y[batch]).abs()/scale).mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   if time.perf_counter()-started>=budget:complete=False;break
  model.eval();parts=[]
  with torch.no_grad():
   for batch in X['validation'].split(batch_size):
    with torch.autocast(device_type=a.device,dtype=torch.bfloat16,enabled=cuda):pred=model(batch)
    parts.append((pred.float()*sd+mu).cpu().numpy())
  pred=np.concatenate(parts);scores=score_arrays(pred,truth['validation']);sync();elapsed=time.perf_counter()-started;passed=complete and qualifies(scores) and elapsed<=budget;entry=dict(epoch=epoch,seconds=elapsed,metrics=scores,complete_epoch=complete);history.append(entry);print(json.dumps(entry),flush=True)
  if passed or elapsed>=budget or a.smoke:break
 a.output.parent.mkdir(parents=True,exist_ok=True);pred_path=a.output.with_suffix('.predictions.csv');checkpoint=a.output.with_suffix('.pt')
 with pred_path.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['id','S1_eV','T1_eV','log_f']);w.writerows([r['id'],*map(float,v)] for r,v in zip(rows['validation'],pred))
 torch.save(dict(model=model.state_dict(),mean=mu.cpu(),std=sd.cpu(),seed=a.seed),checkpoint);pred_hash=digest(pred_path);checkpoint_hash=digest(checkpoint);sync();elapsed=time.perf_counter()-started;passed=passed and elapsed<=budget
 report=dict(protocol=PROTOCOL['id'],protocol_sha256=digest(ROOT/'protocol.json'),manifest_sha256=digest(ROOT/'data-manifest.json'),seed=a.seed,status='smoke' if a.smoke else ('qualified' if passed else 'failed'),total_seconds=elapsed,preprocessing_seconds=prep,history=history,metrics=scores,gpu=torch.cuda.get_device_name() if cuda else None,gpu_count=1 if cuda else 0,cpu_threads=4,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if cuda else 0,slurm_memory_mb=os.environ.get('SLURM_MEM_PER_NODE'),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,commit=commit,source_sha256={k:digest(ROOT/k) for k in ['race.py','model.py','evaluate.py']},predictions_file=pred_path.name,predictions_sha256=pred_hash,checkpoint_file=checkpoint.name,checkpoint_sha256=checkpoint_hash,command=' '.join(__import__('sys').argv),timing='Before CSV reads through final evaluation, checkpoint/prediction exports and hashes; imports/CUDA context/report JSON excluded')
 a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=report['status'],seconds=elapsed,output=str(a.output))),flush=True)
if __name__=='__main__':main()
