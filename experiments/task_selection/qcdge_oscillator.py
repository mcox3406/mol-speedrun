"""S1 transition intensity controls on the exact QCDGE energy-study split."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from screen import SEED,metrics,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.directory;threadpool_limits(4);torch.set_num_threads(4)
 manifest=json.loads((a.out/'qcdge-100000-indices.json').read_text());full=np.load(root/'structure-features.npz');wanted=set(manifest['train']+manifest['validation']);pool=[i for i,k in enumerate(full['ids']) if k in wanted];d={k:full[k][pool] for k in full.files};mapping={k:i for i,k in enumerate(d['ids'])};ix={name:np.asarray([mapping[k] for k in ids]) for name,ids in manifest.items()};tr,va,fit,tune=[ix[n] for n in ['train','validation','inner_train','inner_validation']]
 labels={r['id']:float(r['f_S1']) for r in csv.DictReader((root/'excitation-labels.csv').open())};raw_y=np.asarray([labels[k] for k in d['ids']],dtype=np.float32);assert np.isfinite(raw_y).all() and np.all(raw_y>=0);y=np.log10(1+raw_y/.001)
 report=dict(script_sha256=sha(Path(__file__)),split_manifest_sha256=sha(a.out/'qcdge-100000-indices.json'),transform='log10(1+f/0.001)',notes='Exploratory intensity metric; floor not frozen. Same chemical split as energy screen. CPU cached-feature timings. No quantum labels used as input features.',counts={k:len(v) for k,v in ix.items()},label_audit=dict(quantiles=np.quantile(raw_y,[0,.01,.5,.9,.99,1]).tolist(),zero_fraction=float(np.mean(raw_y==0))),results=[])
 def record(name,pred,**extra):
  raw=np.maximum(0,.001*(10**np.clip(pred,-6,6)-1));r=dict(model=name,transformed=metrics(y[va],pred),raw_f=metrics(raw_y[va],raw),brightness_strata={str(t):dict(n=int(np.sum(raw_y[va]>=t)),mae=float(np.abs(raw[raw_y[va]>=t]-raw_y[va][raw_y[va]>=t]).mean())) for t in [.001,.01,.1]},**extra);report['results'].append(r);(a.out/'qcdge-oscillator.json').write_text(json.dumps(report,indent=2)+'\n');print(r,flush=True)
 record('training_median',np.repeat(np.median(y[tr]),len(va)));record('always_dark',np.zeros(len(va)))
 desc=d['desc'];fp=sparse.csr_matrix(d['fp'],dtype=np.float32);raw=np.hstack([d['fp'],desc]).astype(np.float32)
 configs=[('morgan_ridge',fp,lambda a:Ridge(alpha=a,solver='lsqr'),[10,100]),('descriptor_boosting',desc,lambda a:HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,l2_regularization=a,early_stopping=False,random_state=SEED),[1,10]),('morgan_descriptors_forest',raw,lambda a:ExtraTreesRegressor(n_estimators=64,max_features='sqrt',min_samples_leaf=a,n_jobs=4,random_state=SEED),[1,5])]
 for name,X,factory,choices in configs:
  start=time.monotonic();scores=[]
  for v in choices:
   m=factory(v);m.fit(X[fit],y[fit]);scores.append(float(np.abs(m.predict(X[tune])-y[tune]).mean()))
  best=choices[int(np.argmin(scores))];search=time.monotonic()-start;start=time.monotonic();m=factory(best);m.fit(X[tr],y[tr]);record(name,m.predict(X[va]),chosen_hyperparameter=best,inner_maes=scores,search_seconds=search,fit_eval_seconds=time.monotonic()-start)
 def neural(train,valid,epochs,seed):
  torch.manual_seed(seed);X=raw.copy();mu=X[train,2048:].mean(0);sd=np.maximum(X[train,2048:].std(0),1e-6);X[:,2048:]=(X[:,2048:]-mu)/sd;X=torch.from_numpy(X);mean=y[train].mean();std=y[train].std();Y=torch.from_numpy((y-mean)/std)
  model=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);curve=[];start=time.monotonic()
  for epoch in range(1,epochs+1):
   model.train()
   for batch in torch.tensor(train)[torch.randperm(len(train))].split(512):
    opt.zero_grad();loss=(model(X[batch]).squeeze(1)-Y[batch]).abs().mean();loss.backward();opt.step()
   model.eval()
   with torch.no_grad():pred=torch.cat([model(X[batch]).squeeze(1)*std+mean for batch in torch.tensor(valid).split(2048)]).numpy()
   curve.append(dict(epoch=epoch,mae=float(np.abs(pred-y[valid]).mean())))
  return pred,curve,time.monotonic()-start
 _,curve,elapsed=neural(fit,tune,30,SEED);best=min(curve,key=lambda r:r['mae'])['epoch'];report['inner_curve']=curve;report['selected_epochs']=best;report['search_seconds']=elapsed
 for seed in [SEED,SEED+1,SEED+2]:
  pred,_,elapsed=neural(tr,va,best,seed);record('morgan_descriptors_mlp',pred,training_seed=seed,fit_eval_seconds=elapsed)
if __name__=='__main__':main()
