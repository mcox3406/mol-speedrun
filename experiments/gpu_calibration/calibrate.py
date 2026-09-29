"""One-GPU SMILES regression calibration; train-only chemical model selection."""
import argparse,csv,hashlib,json,os,platform,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Block(nn.Module):
 def __init__(self,width,heads,dropout):
  super().__init__();self.heads=heads;self.norm1=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.proj=nn.Linear(width,width);self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width));self.dropout=dropout
 def forward(self,x,mask):
  b,t,w=x.shape;q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,w//self.heads).permute(2,0,3,1,4).unbind(0)
  h=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=self.dropout if self.training else 0.).transpose(1,2).reshape(b,t,w)
  x=x+F.dropout(self.proj(h),p=self.dropout,training=self.training)
  return x+F.dropout(self.ff(self.norm2(x)),p=self.dropout,training=self.training)

class Regressor(nn.Module):
 def __init__(self,width=192,layers=4,heads=6,dropout=.1):
  super().__init__();self.token=nn.Embedding(129,width,padding_idx=0);self.pos=nn.Embedding(512,width);self.blocks=nn.ModuleList([Block(width,heads,dropout) for _ in range(layers)]);self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,3)
 def forward(self,tokens):
  mask=tokens.ne(0);x=self.token(tokens)+self.pos(torch.arange(tokens.shape[1],device=tokens.device))
  for block in self.blocks:x=block(x,mask)
  x=self.norm(x);return self.head((x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True))

def four(v):return torch.stack([v[:,0],v[:,1],v[:,0]-v[:,1],v[:,2]],dim=1)

def intensity_metrics(pred_log,true_f):
 raw=torch.clamp(.001*(torch.pow(10,pred_log.clamp(-6,6))-1),min=0);bright=true_f>=.1
 return dict(mae=float((raw-true_f).abs().mean()),bright_n=int(bright.sum()),bright_mae=float((raw[bright]-true_f[bright]).abs().mean()) if bright.any() else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=20260929);p.add_argument('--tune-seconds',type=float,default=600);p.add_argument('--refit-seconds',type=float,default=720);p.add_argument('--max-epochs',type=int,default=200);p.add_argument('--patience',type=int,default=20);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--width',type=int,default=192);p.add_argument('--layers',type=int,default=4);p.add_argument('--heads',type=int,default=6);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 torch.set_num_threads(4);cuda=args.device=='cuda'
 if cuda:
  assert torch.cuda.is_available();torch.cuda.init();torch.cuda.synchronize();assert torch.cuda.is_bf16_supported();torch.set_float32_matmul_precision('high')
 def sync():
  if cuda:torch.cuda.synchronize()
 start=time.perf_counter();manifest=json.loads((args.data/'manifest.json').read_text());split=json.loads(args.split.read_text());assert digest(args.split)==manifest['split_manifest_sha256']
 rows=[]
 for part in ['train','validation']:
  path=args.data/manifest['files'][part]['path'];assert digest(path)==manifest['files'][part]['sha256'];rr=list(csv.DictReader(path.open()));assert [r['id'] for r in rr]==split[part];rows+=rr
 mapping={r['id']:i for i,r in enumerate(rows)};indices={k:torch.tensor([mapping[v] for v in ids],device=args.device) for k,ids in split.items()}
 if args.smoke:indices={k:v[:min(len(v),128)] for k,v in indices.items()}
 codes=[[ord(c)+1 for c in r['smiles']] for r in rows];assert max(map(len,codes))<=512 and all(max(v)<129 for v in codes)
 x=torch.zeros(len(rows),max(map(len,codes)),dtype=torch.long)
 for i,v in enumerate(codes):x[i,:len(v)]=torch.tensor(v)
 X=x.to(args.device);y=torch.tensor([[float(r['S1_eV']),float(r['T1_eV']),float(r['f_S1'])] for r in rows],device=args.device);raw_f=y[:,2].clone();y[:,2]=torch.log10(1+y[:,2]/.001);Y=four(y);sync();preprocessing=time.perf_counter()-start
 names=['S1_eV','T1_eV','delta_ST_eV','log10_1_plus_f_over_0001']
 @torch.no_grad()
 def evaluate(model,idx,mu,sd):
  model.eval();parts=[]
  for batch in idx.split(args.batch_size):
   with torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=cuda):pred=model(X[batch])
   parts.append(pred.float()*sd+mu)
  pred=torch.cat(parts);truth=y[idx];p4=four(pred);t4=four(truth);error=(p4-t4).abs().mean(0);mse=(p4-t4).square().mean(0);variance=(t4-t4.mean(0)).square().mean(0)
  scores={n:dict(mae=float(error[j]),rmse=float(mse[j].sqrt()),r2=float(1-mse[j]/variance[j])) for j,n in enumerate(names)}
  scores['raw_f']=intensity_metrics(pred[:,2],raw_f[idx])
  return scores,error
 def fit(train,evaluate_ix,epochs,budget,select):
  phase_start=time.perf_counter()
  torch.manual_seed(args.seed);mu=y[train].mean(0);sd=y[train].std(0,unbiased=False);scale=Y[train].std(0,unbiased=False);model=Regressor(args.width,args.layers,args.heads).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01,fused=cuda);sync();history=[];best=float('inf');chosen=0;completed=0
  for epoch in range(1,epochs+1):
   model.train();train_loss=0.;seen=0;order=train[torch.randperm(len(train),device=args.device)]
   for batch in order.split(args.batch_size):
    with torch.autocast(device_type=args.device,dtype=torch.bfloat16,enabled=cuda):pred=model(X[batch])
    loss=((four(pred.float()*sd+mu)-Y[batch]).abs()/scale).mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();train_loss+=float(loss.detach())*len(batch);seen+=len(batch)
    if time.perf_counter()-phase_start>=budget:break
   sync();complete=seen==len(train)
   if complete:completed=epoch
   entry=dict(epoch=epoch,complete_epoch=complete,seconds=time.perf_counter()-phase_start,train_normalized_mae=train_loss/seen)
   if select:
    scores,error=evaluate(model,evaluate_ix,mu,sd);criterion=float((error/scale).mean());entry.update(inner_scores=scores,inner_normalized_mae=criterion)
    if complete and criterion<best:best=criterion;chosen=epoch
   history.append(entry);print(json.dumps(dict(phase='tune' if select else 'refit',**entry)),flush=True)
   if time.perf_counter()-phase_start>=budget or not complete:break
   if select and chosen and epoch-chosen>=args.patience:break
  sync();return model,mu,sd,history,chosen,completed,time.perf_counter()-phase_start
 model,mu,sd,inner,best,completed,tune_seconds=fit(indices['inner_train'],indices['inner_validation'],args.max_epochs,args.tune_seconds,True)
 if not best:raise RuntimeError('Tuning budget did not complete one epoch')
 del model
 model,mu,sd,history,_,completed,refit_seconds=fit(indices['train'],None,best,args.refit_seconds,False)
 sync();evaluation_start=time.perf_counter();scores,_=evaluate(model,indices['validation'],mu,sd);sync();evaluation_seconds=time.perf_counter()-evaluation_start
 args.output.parent.mkdir(parents=True,exist_ok=True);checkpoint=args.output.with_suffix('.pt');torch.save(dict(model=model.state_dict(),mean=mu.cpu(),std=sd.cpu(),width=args.width,layers=args.layers,heads=args.heads),checkpoint)
 report=dict(status='cpu_smoke_only' if args.smoke else ('completed' if completed==best else 'refit_budget_exhausted'),model='bidirectional_SMILES_transformer',parameters=sum(v.numel() for v in model.parameters()),seed=args.seed,torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),gpu=torch.cuda.get_device_name() if cuda else None,device=args.device,slurm_job_id=os.environ.get('SLURM_JOB_ID'),source_sha256=digest(Path(__file__)),manifest_sha256=digest(args.data/'manifest.json'),split_sha256=digest(args.split),counts={k:len(v) for k,v in indices.items()},max_sequence_length=X.shape[1],configuration=dict(width=args.width,layers=args.layers,heads=args.heads,batch_size=args.batch_size,lr=.0003,dropout=.1),preprocessing_seconds=preprocessing,tuning_seconds=tune_seconds,selected_epochs=best,completed_refit_epochs=completed,refit_seconds=refit_seconds,evaluation_seconds=evaluation_seconds,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if cuda else None,inner_history=inner,refit_history=history,validation=scores,notes='Fixed chemical split. Epoch count selected exclusively on inner chemical holdout, then model freshly refit on all training data. No outer validation checkpoint selection. CUDA context initialization and environment startup excluded; no compile. Preprocessing and tuning/refit/evaluation clocks reported separately; calibration, not an eligible speedrun record.')
 args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=report['status'],output=str(args.output),validation=scores)),flush=True)
if __name__=='__main__':main()
