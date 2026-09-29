"""QCDGE development baselines; all selection uses training-only chemical groups."""
import argparse,csv,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from rdkit import DataStructs
from screen import SEED,group_split,metrics,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--sample',type=int,default=100000);a=p.parse_args();root=a.directory;a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);threadpool_limits(4)
 audit=json.loads((root/f'extraction-audit-{a.sample}.json').read_text())
 selected=set(json.loads((root/f'extraction-sample-{a.sample}.json').read_text()));labels={r['id']:r for r in csv.DictReader((root/'excitation-labels.csv').open())};failed={r['id'] for r in audit['errors']};assert selected<=set(labels)|failed,'Incomplete extraction'
 full=np.load(root/'structure-features.npz');pool=np.asarray([i for i,k in enumerate(full['ids']) if k in selected and k in labels]);d={k:full[k][pool] for k in full.files};ids=d['ids'];y=np.asarray([[float(labels[k]['S1_eV']),float(labels[k]['T1_eV'])] for k in ids],dtype=np.float32)
 bad=(y<=0).any(axis=1)|np.asarray([float(labels[k]['f_S1'])<0 for k in ids]);qc_excluded=[dict(id=str(ids[i]),S1_eV=float(y[i,0]),T1_eV=float(y[i,1]),f_S1=float(labels[ids[i]]['f_S1'])) for i in np.flatnonzero(bad)]
 d={k:v[~bad] for k,v in d.items()};ids=d['ids'];y=y[~bad]
 def three(v):return np.column_stack([v,v[:,0]-v[:,1]])
 Y=three(y);names=['S1_eV','T1_eV','delta_ST_eV'];groups=d['groups'];tr,va=group_split(groups,.2,SEED);ia,ib=group_split(groups[tr],.15,SEED+1);fit,tune=tr[ia],tr[ib]
 for field in ['groups','scaffolds','families','smiles']:
  assert not set(d[field][tr])&set(d[field][va]),field
  assert not set(d[field][fit])&set(d[field][tune]),field
 report=dict(seed=SEED,script_sha256=sha(Path(__file__)),source_archive_md5=json.loads((root/'final_all.hdf5.verified.json').read_text())['md5'],sample=a.sample,counts=dict(eligible_labeled=len(ids),train=len(tr),validation=len(va),inner_train=len(fit),inner_validation=len(tune)),label_audit=dict(qc_excluded=qc_excluded,qc_rule='Require positive S1 and T1 and nonnegative S1 oscillator strength; retain small positive energies and gaps without threshold cuts.',nonpositive_S1=int(np.sum(y[:,0]<=0)),nonpositive_T1=int(np.sum(y[:,1]<=0)),negative_gap=int(np.sum(Y[:,2]<0)),target_quantiles=np.quantile(Y,[0,.01,.5,.99,1],axis=0).tolist()),distribution={},results=[],notes='Exploratory development split, one chemical partition, no final test. Full-dataset chemical groups built without labels; uniformly sampled extraction pool. Models predict S1/T1 and gap is their difference. CPU timings exclude cached features and are not eligible speedrun times.')
 for name,ix in [('train',tr),('validation',va)]:
  report['distribution'][name]=dict(groups=len(set(groups[ix])),largest_group=max(Counter(groups[ix]).values()),heavy_atom_quantiles=np.quantile(d['desc'][ix,6],[.05,.5,.95]).tolist(),source_counts=dict(Counter(str(k)[:2] for k in ids[ix])),target_mean=Y[ix].mean(0).tolist(),target_std=Y[ix].std(0).tolist(),acyclic_fraction=float(np.mean([s.startswith('chain:') for s in d['scaffolds'][ix]])))
 out=a.out/f'qcdge-{a.sample}.json';(a.out/f'qcdge-{a.sample}-indices.json').write_text(json.dumps({name:ids[ix].tolist() for name,ix in [('train',tr),('validation',va),('inner_train',fit),('inner_validation',tune)]}))
 def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 def record(name,pred,**extra):
  r=dict(model=name,targets={n:metrics(Y[va,j],three(pred)[:,j]) for j,n in enumerate(names)},**extra);report['results'].append(r);save();print(r,flush=True)
 def selection(pred):return float((np.abs(three(pred)-Y[tune])/Y[fit].std(0)).mean())
 record('training_median',np.tile(np.median(y[tr],axis=0),(len(va),1)))
 desc=d['desc'];fp=sparse.csr_matrix(d['fp'],dtype=np.float32)
 configs=[('atom_counts_ridge',desc[:,:6],lambda v:make_pipeline(StandardScaler(),Ridge(alpha=v,solver='svd')),[1,100]),('morgan_ridge',fp,lambda v:Ridge(alpha=v,solver='lsqr'),[10,100])]
 for name,X,factory,choices in configs:
  start=time.monotonic();scores=[]
  for v in choices:
   m=factory(v);m.fit(X[fit],y[fit]);scores.append(selection(m.predict(X[tune])))
  best=choices[int(np.argmin(scores))];search=time.monotonic()-start;start=time.monotonic();m=factory(best);m.fit(X[tr],y[tr]);record(name,m.predict(X[va]),hyperparameter=best,inner_scores=scores,search_seconds=search,fit_eval_seconds=time.monotonic()-start)
 def boost(train,valid,reg):
  pred=[]
  for j in range(2):
   m=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,l2_regularization=reg,early_stopping=False,random_state=SEED);m.fit(desc[train],y[train,j]);pred.append(m.predict(desc[valid]))
  return np.column_stack(pred)
 start=time.monotonic();choices=[1,10];scores=[selection(boost(fit,tune,v)) for v in choices];best=choices[int(np.argmin(scores))];search=time.monotonic()-start;start=time.monotonic();pred=boost(tr,va,best);record('descriptor_boosting',pred,hyperparameter=best,inner_scores=scores,search_seconds=search,fit_eval_seconds=time.monotonic()-start)
 raw=np.hstack([d['fp'],desc]).astype(np.float32)
 # Joint tree control, standardized training targets to weight S1 and T1 equally.
 def forest(train,valid,leaf):
  mean=y[train].mean(0);std=y[train].std(0);m=ExtraTreesRegressor(n_estimators=64,max_features='sqrt',min_samples_leaf=leaf,n_jobs=4,random_state=SEED);m.fit(raw[train],(y[train]-mean)/std);return m.predict(raw[valid])*std+mean
 start=time.monotonic();choices=[1,5];scores=[selection(forest(fit,tune,v)) for v in choices];best=choices[int(np.argmin(scores))];search=time.monotonic()-start;start=time.monotonic();pred=forest(tr,va,best);record('morgan_descriptors_forest',pred,hyperparameter=best,inner_scores=scores,search_seconds=search,fit_eval_seconds=time.monotonic()-start)
 def neural(train,valid,epochs,seed):
  torch.manual_seed(seed);X=raw.copy();mean=X[train,2048:].mean(0);std=np.maximum(X[train,2048:].std(0),1e-6);X[:,2048:]=(X[:,2048:]-mean)/std;X=torch.from_numpy(X)
  mu=torch.from_numpy(y[train].mean(0));sd=torch.from_numpy(y[train].std(0));target=torch.from_numpy(Y);scale=torch.from_numpy(Y[train].std(0));model=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,2));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);curve=[];start=time.monotonic()
  for epoch in range(1,epochs+1):
   model.train()
   for ix in torch.tensor(train)[torch.randperm(len(train))].split(512):
    opt.zero_grad();p=model(X[ix])*sd+mu;p=torch.cat([p,p[:,0:1]-p[:,1:2]],dim=1);loss=((p-target[ix]).abs()/scale).mean();loss.backward();opt.step()
   model.eval()
   with torch.no_grad():pred=torch.cat([model(X[ix])*sd+mu for ix in torch.tensor(valid).split(2048)]).numpy()
   errors=np.abs(three(pred)-Y[valid]).mean(0);curve.append(dict(epoch=epoch,mae=errors.tolist(),normalized_mae=float((errors/scale.numpy()).mean())))
   if epoch%10==0:print('MLP',seed,epoch,curve[-1],flush=True)
  return pred,curve,time.monotonic()-start
 _,curve,elapsed=neural(fit,tune,30,SEED);best=min(curve,key=lambda v:v['normalized_mae'])['epoch'];report['mlp_inner_curve']=curve;report['mlp_selected_epochs']=best;report['mlp_search_seconds']=elapsed
 for seed in [SEED,SEED+1,SEED+2]:
  pred,_,elapsed=neural(tr,va,best,seed);record('morgan_descriptors_mlp',pred,training_seed=seed,selected_epochs=best,fit_eval_seconds=elapsed)
 fps=[DataStructs.CreateFromBinaryText(row.tobytes()) for row in np.packbits(d['fp'],axis=1,bitorder='little')];trainfps=[fps[i] for i in tr];query=np.random.default_rng(SEED).choice(va,min(500,len(va)),replace=False);similarities=[];neighbors=[]
 for q in query:
  sim=np.asarray(DataStructs.BulkTanimotoSimilarity(fps[q],trainfps));similarities.append(float(sim.max()));neighbors.append(int(tr[sim.argmax()]))
 report['nearest_train_similarity']=dict(n_queries=len(query),quantiles=np.quantile(similarities,[.1,.5,.9]).tolist(),fraction_ge_08=float(np.mean(np.asarray(similarities)>=.8)))
 report['nearest_neighbor_subset']={n:metrics(Y[query,j],Y[neighbors,j]) for j,n in enumerate(names)};save();print('DONE',out,flush=True)
if __name__=='__main__':main()
