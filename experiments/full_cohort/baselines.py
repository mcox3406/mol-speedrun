"""Joint full-cohort CPU controls; hyperparameters selected on inner chemical holdout."""
import argparse, hashlib, json, time, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def four(y):return np.column_stack([y[:,:2],y[:,0]-y[:,1],y[:,2]])
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();threadpool_limits(4);torch.set_num_threads(4)
 d=np.load(a.data);lookup={k:i for i,k in enumerate(d['ids'])};ix={k:np.asarray([lookup[v] for v in ids]) for k,ids in json.loads(a.split.read_text()).items()};tr,va,fit,tune=[ix[k] for k in ['train','validation','inner_train','inner_validation']];y=d['y'].copy();raw_f=y[:,2].copy();y[:,2]=np.log10(1+y[:,2]/.001)
 report=dict(status='running',source_sha256=sha(Path(__file__)),data_sha256=sha(a.data),split_sha256=sha(a.split),counts={k:len(v) for k,v in ix.items()},python=platform.python_version(),results=[],notes='Joint S1, T1, transformed oscillator strength controls. Gap derived. Inner chemical model selection; outer validation evaluated after refit. Cached feature timings are not speedrun records.');a.output.parent.mkdir(parents=True,exist_ok=True)
 def save():a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 def score(pred,idx):return float((np.abs(four(pred)-four(y[idx]))/four(y[fit]).std(0)).mean())
 def record(name,pred,**extra):
  P=four(pred);Y=four(y[va]);metrics={k:dict(mae=float(np.abs(P[:,j]-Y[:,j]).mean()),r2=float(1-np.square(P[:,j]-Y[:,j]).sum()/np.square(Y[:,j]-Y[:,j].mean()).sum())) for j,k in enumerate(['S1_eV','T1_eV','delta_ST_eV','log10_1_plus_f_over_0001'])};raw=np.maximum(0,.001*(10**np.clip(pred[:,2],-6,6)-1));bright=raw_f[va]>=.1;metrics['raw_f']=dict(mae=float(np.abs(raw-raw_f[va]).mean()),bright_n=int(bright.sum()),bright_mae=float(np.abs(raw[bright]-raw_f[va][bright]).mean()));r=dict(model=name,validation=metrics,**extra);report['results'].append(r);save();print(json.dumps(r),flush=True)
 record('training_median',np.tile(np.median(y[tr],axis=0),(len(va),1)))
 fp=sparse.csr_matrix(d['fp'],dtype=np.float32);desc=d['desc'];raw=np.column_stack([d['fp'],desc]).astype(np.float32)
 for name,X,factory,choices in [('atom_counts_ridge',desc[:,:6],lambda v:Ridge(alpha=v,solver='lsqr'),[10,100]),('morgan_ridge',fp,lambda v:Ridge(alpha=v,solver='lsqr'),[10,100]),('morgan_descriptors_forest',raw,lambda v:ExtraTreesRegressor(n_estimators=64,max_features='sqrt',min_samples_leaf=v,n_jobs=4,random_state=20260929),[1,5])]:
  start=time.monotonic();scores=[]
  for choice in choices:
   mu=y[fit].mean(0);sd=y[fit].std(0);m=factory(choice);m.fit(X[fit],(y[fit]-mu)/sd);scores.append(score(m.predict(X[tune])*sd+mu,tune));del m
  chosen=choices[int(np.argmin(scores))];search=time.monotonic()-start;start=time.monotonic();mu=y[tr].mean(0);sd=y[tr].std(0);m=factory(chosen);m.fit(X[tr],(y[tr]-mu)/sd);record(name,m.predict(X[va])*sd+mu,selected_hyperparameter=chosen,inner_scores=scores,search_seconds=search,fit_eval_seconds=time.monotonic()-start);del m
 del fp
 def neural(train,valid,epochs,seed,select):
  start=time.monotonic();torch.manual_seed(seed);X=torch.from_numpy(raw.copy());mu=X[train,2048:].mean(0);sd=X[train,2048:].std(0,unbiased=False).clamp_min(1e-6);X[:,2048:]=(X[:,2048:]-mu)/sd;Y=torch.from_numpy(y);mean=Y[train].mean(0);std=Y[train].std(0,unbiased=False);scale=torch.from_numpy(four(y[train]).std(0));target=torch.from_numpy(four(y));m=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,3));opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);curve=[]
  for epoch in range(1,epochs+1):
   m.train()
   for batch in torch.tensor(train)[torch.randperm(len(train))].split(512):
    pred=m(X[batch])*std+mean;out=torch.stack([pred[:,0],pred[:,1],pred[:,0]-pred[:,1],pred[:,2]],1);loss=((out-target[batch]).abs()/scale).mean();opt.zero_grad();loss.backward();opt.step()
   if select or epoch==epochs:
    m.eval()
    with torch.no_grad():pred=torch.cat([m(X[b])*std+mean for b in torch.tensor(valid).split(2048)]).numpy()
    curve.append(dict(epoch=epoch,normalized_mae=score(pred,valid)));print(json.dumps(dict(phase='mlp_inner' if select else 'mlp_refit',seed=seed,**curve[-1])),flush=True)
  return pred,curve,time.monotonic()-start
 _,curve,elapsed=neural(fit,tune,30,20260929,True);best=min(curve,key=lambda x:x['normalized_mae'])['epoch'];report.update(mlp_inner_curve=curve,mlp_selected_epochs=best,mlp_search_seconds=elapsed);save()
 for seed in [20260929,20260930,20260931]:
  pred,_,elapsed=neural(tr,va,best,seed,False);record('morgan_descriptors_mlp',pred,seed=seed,fit_eval_seconds=elapsed)
 report['status']='completed';save()
if __name__=='__main__':main()
