"""Exploratory Tartarus emitter controls; whole generic scaffold holdout.
Run from experiments/task_selection using the project's Python environment.
"""
import argparse, csv, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import sparse
from threadpoolctl import threadpool_limits
from screen import SEED, group_split, metrics, sha
from audit_groups import generic_topology

def main():
    p=argparse.ArgumentParser(); p.add_argument('--csv',type=Path,required=True); p.add_argument('--work',type=Path,required=True); p.add_argument('--sample',type=int,default=100000); p.add_argument('--split',choices=['generic','typed'],default='generic'); args=p.parse_args()
    args.work.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); threadpool_limits(4)
    cache=args.work/f'emitters-{args.sample}.npz'; start=time.perf_counter()
    if not cache.exists():
        rows=list(csv.DictReader(args.csv.open())); chosen=np.sort(np.random.default_rng(SEED).choice(len(rows),min(args.sample,len(rows)),replace=False))
        fps=[]; ds=[]; ys=[]; groups=[]; smiles=[]; ids=[]; seen=set(); invalid=duplicates=0
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        for k,i in enumerate(chosen):
            r=rows[i]; m=Chem.MolFromSmiles(r['smiles'])
            if m is None: invalid+=1; continue
            smi=Chem.MolToSmiles(m)
            if smi in seen: duplicates+=1; continue
            seen.add(smi); scaffold=MurckoScaffold.GetScaffoldForMol(m)
            # Remove terminal scaffold atoms retained for exocyclic double bonds:
            # an oxo substituent must not separate otherwise identical ring cores.
            sc=Chem.RWMol(scaffold)
            while True:
                leaves=[a.GetIdx() for a in sc.GetAtoms() if a.GetDegree()==1]
                if not leaves: break
                for ix in reversed(leaves): sc.RemoveAtom(ix)
            group=generic_topology(sc if sc.GetNumAtoms() else m)
            counts=Counter(a.GetSymbol() for a in Chem.AddHs(m).GetAtoms())
            ds.append([counts[s] for s in ['H','C','N','O','F','S']]+[m.GetNumHeavyAtoms(),Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m),Descriptors.NumHDonors(m),Descriptors.NumHAcceptors(m),Descriptors.NumRotatableBonds(m),Descriptors.RingCount(m),Descriptors.FractionCSP3(m),Descriptors.BertzCT(m)])
            y=[float(r['singlet-triplet value']),float(r['oscillator strength'])]
            assert np.isfinite(y).all() and min(y)>0
            fps.append(gen.GetFingerprintAsNumPy(m));ys.append(y);groups.append(group);smiles.append(smi);ids.append(int(i))
            if k%10000==0: print('features',k,round(time.perf_counter()-start),flush=True)
        np.savez_compressed(cache,fp=np.asarray(fps,dtype=np.uint8),desc=np.asarray(ds,dtype=np.float32),y=np.asarray(ys,dtype=np.float32),groups=np.asarray(groups),smiles=np.asarray(smiles),ids=ids,metadata=json.dumps(dict(source_rows=len(rows),sampled=len(chosen),invalid=invalid,canonical_duplicates_removed=duplicates,sha256=sha(args.csv),rdkit=rdBase.rdkitVersion)))
    d=np.load(cache); groups=d['groups'] if args.split=='generic' else np.asarray([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(s)),includeChirality=False) for s in d['smiles']]); tr,va=group_split(groups,.2,SEED);ia,ib=group_split(groups[tr],.15,SEED+1); fit,tune=tr[ia],tr[ib]
    assert not set(groups[tr]) & set(groups[va]); assert not set(groups[fit]) & set(groups[tune])
    assert not set(d['smiles'][tr]) & set(d['smiles'][va])
    y=np.column_stack([d['y'][:,0],np.log10(1+d['y'][:,1]/.001)]).astype(np.float32)
    names=['delta_ST_eV','log10_1_plus_f_over_0.001']; desc=d['desc']; fp=sparse.csr_matrix(d['fp'],dtype=np.float32)
    out=args.work/f'emitters-{args.sample}-{args.split}.json'
    report=dict(seed=SEED,data=json.loads(str(d['metadata'])),script_sha256=sha(Path(__file__)),counts=dict(train=len(tr),validation=len(va),inner_train=len(fit),inner_validation=len(tune)),split=('Whole generic ring/linker topology; atom types, bond orders, stereochemistry and terminal scaffold leaves removed; full generic topology for acyclic molecules.' if args.split=='generic' else 'Whole typed nonchiral Bemis-Murcko scaffold groups.'),notes='Exploratory development split, not final test. Uniform molecule sample precedes whole-group partition. One seed. Cached-feature CPU timings are not speedrun times. f floor=0.001 is exploratory, not a frozen metric.',distribution={},results=[])
    for name,ix in [('train',tr),('validation',va)]:
        report['distribution'][name]=dict(groups=len(set(groups[ix])),largest_group=max(Counter(groups[ix]).values()),heavy_atom_quantiles=np.quantile(desc[ix,6],[.05,.5,.95]).tolist(),target_mean=y[ix].mean(0).tolist(),target_std=y[ix].std(0).tolist(),fraction_f_below_0001=float(np.mean(d['y'][ix,1]<.001)))
    (args.work/f'emitters-{args.sample}-{args.split}-indices.json').write_text(json.dumps({name:d['ids'][ix].tolist() for name,ix in [('train',tr),('validation',va),('inner_train',fit),('inner_validation',tune)]}))
    def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def score(name,j,pred,**extra):
        r=dict(model=name,target=names[j],**metrics(y[va,j],pred),**extra)
        if j==1:
            raw=np.maximum(0,.001*(10**np.clip(pred,-6,6)-1)); r['raw_f']=metrics(d['y'][va,1],raw)
            r['brightness_strata']={str(t):dict(n=int(np.sum(d['y'][va,1]>=t)),mae=float(np.abs(raw[d['y'][va,1]>=t]-d['y'][va,1][d['y'][va,1]>=t]).mean())) for t in [.001,.01,.1]}
        report['results'].append(r);save();print(r,flush=True)
    for j in range(2):
        score('training_median',j,np.repeat(np.median(y[tr,j]),len(va)))
        if j==1: score('always_dark',j,np.zeros(len(va)))
        configs=[('counts_ridge',desc[:,:6],lambda a:make_pipeline(StandardScaler(),Ridge(alpha=a)),[1,100]),('morgan_ridge',fp,lambda a:Ridge(alpha=a,solver='lsqr'),[10,100]),('descriptors_boosting',desc,lambda a:HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,l2_regularization=a,early_stopping=False,random_state=SEED),[1,10])]
        for name,X,factory,choices in configs:
            vals=[];t0=time.perf_counter()
            for a in choices:
                m=factory(a);m.fit(X[fit],y[fit,j]);vals.append(float(np.abs(m.predict(X[tune])-y[tune,j]).mean()))
            best=choices[int(np.argmin(vals))]; search=time.perf_counter()-t0;t0=time.perf_counter();m=factory(best);m.fit(X[tr],y[tr,j]);pred=m.predict(X[va]);score(name,j,pred,alpha=best,inner_maes=vals,search_seconds=search,fit_eval_seconds=time.perf_counter()-t0)
    raw=np.hstack([d['fp'],desc]).astype(np.float32)
    def neural(train,valid,epochs):
        torch.manual_seed(SEED);X=raw.copy();mu=X[train,2048:].mean(0);sd=np.maximum(X[train,2048:].std(0),1e-6);X[:,2048:]=(X[:,2048:]-mu)/sd;X=torch.from_numpy(X)
        mean=y[train].mean(0);std=y[train].std(0);Y=torch.from_numpy((y-mean)/std)
        model=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,2));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);curve=[];t0=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train()
            for ix in torch.tensor(train)[torch.randperm(len(train))].split(512):
                opt.zero_grad();loss=(model(X[ix])-Y[ix]).abs().mean();loss.backward();opt.step()
            model.eval()
            with torch.no_grad():pred=torch.cat([model(X[ix]) for ix in torch.tensor(valid).split(2048)]).numpy()*std+mean
            error=np.abs(pred-y[valid]).mean(0);curve.append(dict(epoch=epoch,mae=error.tolist(),normalized_mae=float((error/std).mean())))
            if epoch%5==0:print('MLP',len(train),epoch,curve[-1],flush=True)
        return curve,pred,time.perf_counter()-t0
    curve,_,seconds=neural(fit,tune,30);best=min(curve,key=lambda a:a['normalized_mae'])['epoch'];final,pred,elapsed=neural(tr,va,best)
    report['mlp_inner_curve']=curve;report['mlp_selected_epochs']=best;report['mlp_search_seconds']=seconds
    for j in range(2):score('morgan_descriptors_mlp',j,pred[:,j],fit_eval_seconds=elapsed)
    fps=[DataStructs.CreateFromBinaryText(row.tobytes()) for row in np.packbits(d['fp'],axis=1,bitorder='little')];trainfps=[fps[i] for i in tr];query=np.random.default_rng(SEED).choice(va,min(300,len(va)),replace=False); sims=[]; neighbors=[]
    for q in query:
        similarities=np.asarray(DataStructs.BulkTanimotoSimilarity(fps[q],trainfps));sims.append(float(similarities.max()));neighbors.append(int(tr[similarities.argmax()]))
    report['nearest_train_similarity']=dict(queries=len(query),quantiles=np.quantile(sims,[.1,.5,.9]).tolist(),fraction_ge_08=float(np.mean(np.asarray(sims)>=.8)))
    report['nearest_neighbor_subset']={name:metrics(y[query,j],y[neighbors,j]) for j,name in enumerate(names)};save()
if __name__=='__main__':main()
