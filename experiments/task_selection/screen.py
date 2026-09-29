"""Exploratory task selection; not a frozen speedrun protocol.

Reproduce: python screen.py --dataset esol --split scaffold
All model selection uses an internal training-only holdout. Evaluation labels
are used only for reporting, but these development splits are NOT final tests.
"""
import argparse, csv, gzip, hashlib, io, json, os, platform, time, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from rdkit import Chem, DataStructs, rdBase, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.warning'); RDLogger.DisableLog('rdApp.error')
ROOT=Path(__file__).resolve().parent
SEED=20260929
PCQM_TRAIN_SIZE=50000
HARTREE_EV=27.211386245988
FEATURE_NAMES=['MW','logP','TPSA','HBD','HBA','rotatable','rings','aromatic_rings','aliphatic_rings','heteroatoms','fraction_sp3','heavy_atoms','valence_electrons','molar_refractivity','BertzCT','formal_charge','conjugated_bonds','aromatic_atoms','H','C','N','O','F','S','Cl','Br','P','I','Si','other_atoms']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def group_split(groups, fraction, seed):
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(np.zeros(len(groups)),groups=groups))
    return a,b

def read_data(dataset):
    path=ROOT/'raw'/f'{dataset}.csv'
    if dataset=='pcqm':
        import torch
        path=ROOT/'raw/pcqm4m-v2.zip'
        with zipfile.ZipFile(path) as z:
            print(z.namelist(),flush=True)
            split_name=next(n for n in z.namelist() if n.endswith('split_dict.pt'))
            # Official archive; load only tensor/primitive data via restricted loader.
            with torch.serialization.safe_globals([(np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'), np.ndarray, np.dtype, type(np.dtype('int64'))]):
                split=torch.load(io.BytesIO(z.read(split_name)),weights_only=True)
            train=np.asarray(split['train']); val=np.asarray(split['valid'])
            rng=np.random.default_rng(SEED)
            chosen_train=rng.choice(train,PCQM_TRAIN_SIZE,replace=False)
            chosen_val=rng.choice(val,10000,replace=False)
            selected={int(i):'train' for i in chosen_train}; selected.update({int(i):'val' for i in chosen_val})
            csvname=next(n for n in z.namelist() if n.endswith('data.csv.gz'))
            rows=[]
            with z.open(csvname) as f, gzip.open(f,'rt') as g:
                for i,r in enumerate(csv.DictReader(g)):
                    if i in selected:
                        rows.append(dict(id=i,smiles=r['smiles'],y=[float(r['homolumogap'])],official=selected[i]))
        return rows,['gap_eV'],path
    source=list(csv.DictReader(path.read_text().splitlines()))
    if dataset=='qm9':
        # Uniform sample, not first rows (QM9 ordering strongly correlates with size).
        indices=np.random.default_rng(SEED).choice(len(source),60000,replace=False)
        rows=[dict(id=int(i),smiles=source[i]['smiles'],y=[float(source[i]['gap'])*HARTREE_EV,float(source[i]['mu']),float(source[i]['u0'])*HARTREE_EV]) for i in sorted(indices)]
        return rows,['gap_eV','dipole_D','total_energy_eV'],path
    target='exp' if dataset=='lipo' else 'measured log solubility in mols per litre'
    return [dict(id=i,smiles=r['smiles'],y=[float(r[target])]) for i,r in enumerate(source)],['logD' if dataset=='lipo' else 'logS'],path

def features(dataset):
    suffix=f'-{PCQM_TRAIN_SIZE}' if dataset=='pcqm' and PCQM_TRAIN_SIZE!=50000 else ''
    cache=ROOT/f'{dataset}{suffix}-features.npz'
    if cache.exists():
        d=np.load(cache,allow_pickle=False); return {k:d[k] for k in d.files}
    start=time.perf_counter(); rows,targets,path=read_data(dataset)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    records=[]; desc=[]; fps=[]; ys=[]; skipped=[]; ids=[]; scaffolds=[]; officials=[]
    seen={}; duplicates=0; symbols=['H','C','N','O','F','S','Cl','Br','P','I','Si']
    for i,r in enumerate(rows):
        mol=Chem.MolFromSmiles(r['smiles'].strip())
        if mol is None: skipped.append(r['id']); continue
        smi=Chem.MolToSmiles(mol,isomericSmiles=True)
        if smi in seen:
            duplicates+=1
            # Keep first occurrence; never pool labels across an official split.
            continue
        seen[smi]=r['id']
        atoms=Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())
        d=[Descriptors.MolWt(mol),Crippen.MolLogP(mol),rdMolDescriptors.CalcTPSA(mol),Lipinski.NumHDonors(mol),Lipinski.NumHAcceptors(mol),Lipinski.NumRotatableBonds(mol),rdMolDescriptors.CalcNumRings(mol),rdMolDescriptors.CalcNumAromaticRings(mol),rdMolDescriptors.CalcNumAliphaticRings(mol),Lipinski.NumHeteroatoms(mol),rdMolDescriptors.CalcFractionCSP3(mol),mol.GetNumHeavyAtoms(),Descriptors.NumValenceElectrons(mol),Crippen.MolMR(mol),Descriptors.BertzCT(mol),Chem.GetFormalCharge(mol),sum(b.GetIsConjugated() for b in mol.GetBonds()),sum(a.GetIsAromatic() for a in mol.GetAtoms())]+[atoms[s] for s in symbols]+[sum(v for k,v in atoms.items() if k not in symbols)]
        assert len(d)==len(FEATURE_NAMES) and np.isfinite(d).all()
        desc.append(d); fps.append(gen.GetFingerprintAsNumPy(mol)); records.append(smi); ys.append(r['y']); ids.append(r['id'])
        scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False))
        officials.append(r.get('official',''))
        if i%10000==0: print(dataset,'features',i,'seconds',round(time.perf_counter()-start),flush=True)
    result=dict(desc=np.asarray(desc,dtype=np.float32),fp=np.asarray(fps,dtype=np.uint8),y=np.asarray(ys),smiles=np.asarray(records),ids=np.asarray(ids),scaffolds=np.asarray(scaffolds),official=np.asarray(officials),targets=np.asarray(targets),metadata=np.asarray(json.dumps(dict(source_sha256=sha(path),rdkit=rdBase.rdkitVersion,sampled_rows=len(rows),retained=len(records),invalid_ids=skipped,duplicates_removed=duplicates,feature_seconds=time.perf_counter()-start,source_path=path.name,seed=SEED))))
    np.savez_compressed(cache,**result); return result

def metrics(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))

def main():
    global PCQM_TRAIN_SIZE
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['esol','lipo','qm9','pcqm'],required=True);p.add_argument('--split',choices=['scaffold','generic','chemical'],required=True); p.add_argument('--forest-trees',type=int,default=64)
    p.add_argument('--train-size',type=int,default=50000)
    p.add_argument('--fit-size',type=int,default=0,help='Nested training subset; evaluation molecules stay fixed')
    p.add_argument('--skip-forest',action='store_true')
    p.add_argument('--closed-shell',action='store_true')
    args=p.parse_args(); PCQM_TRAIN_SIZE=args.train_size; threadpool_limits(4)
    d=features(args.dataset); n=len(d['y']); inds=np.arange(n)
    groups=d['scaffolds']
    if args.split in ('generic','chemical'):
        assert args.dataset=='pcqm' and PCQM_TRAIN_SIZE==200000
        groups=np.load(ROOT/'pcqm-chemical-groups.npz',allow_pickle=False)['groups' if args.split=='generic' else 'hybrid']
    pool=np.flatnonzero(np.load(ROOT/'pcqm-eligible.npy')) if args.closed_shell else np.arange(n)
    if args.closed_shell: assert args.dataset=='pcqm' and PCQM_TRAIN_SIZE==200000
    ta,tb=group_split(groups[pool],.2,SEED);tr,va=pool[ta],pool[tb]
    if args.fit_size:
        tr=np.random.default_rng(SEED+2).permutation(tr)[:args.fit_size]
    # One internal validation partition reused for all model choices.
    inner_group=groups[tr]
    ia,ib=group_split(inner_group,.15,SEED+1); fit=tr[ia]; tune=tr[ib]
    assert not set(d['smiles'][tr]) & set(d['smiles'][va])
    assert not set(groups[tr]) & set(groups[va])
    suffix=f'-{PCQM_TRAIN_SIZE}' if args.dataset=='pcqm' and PCQM_TRAIN_SIZE!=50000 else ''
    if args.fit_size: suffix+=f'-fit{args.fit_size}'
    if args.closed_shell: suffix+='-closed-shell'
    out=ROOT/'results'/f'{args.dataset}-{args.split}{suffix}.json'
    desc=d['desc']; fp=sparse.csr_matrix(d['fp'],dtype=np.float32); joined=np.hstack([d['fp'],desc])
    report=dict(dataset=args.dataset,split=args.split,closed_shell=args.closed_shell,eligible_pool_size=len(pool),seed=SEED,counts=dict(train=len(tr),val=len(va),inner_fit=len(fit),inner_tune=len(tune)),data=json.loads(str(d['metadata'])),script_sha256=sha(Path(__file__)),hardware='Apple M2 Pro CPU; 4 numerical-library threads; 4 forest workers',platform=platform.platform(),forest_trees=args.forest_trees,forest_max_features='sqrt',notes='Exploratory development validation; custom sampled split, not official leaderboard score. Training-only internal tuning. No optimized neural model yet.',features=FEATURE_NAMES,results=[])
    report['distribution']={name:dict(heavy_atom_quantiles=np.quantile(desc[ix,11],[.05,.5,.95]).tolist(),unique_scaffolds=len(set(d['scaffolds'][ix])),acyclic_fraction=float(np.mean(d['scaffolds'][ix]=='')),target_mean=d['y'][ix].mean(0).tolist(),target_std=d['y'][ix].std(0).tolist()) for name,ix in [('train',tr),('val',va)]}
    train_scaffolds=set(d['scaffolds'][tr])
    report['distribution']['val_scaffold_unseen_fraction']=float(np.mean([s not in train_scaffolds for s in d['scaffolds'][va]]))
    # Index manifest permits exact reconstruction without redistributing raw files.
    split_manifest=dict(train_source_ids=d['ids'][tr].tolist(),validation_source_ids=d['ids'][va].tolist(),inner_fit_source_ids=d['ids'][fit].tolist(),inner_tune_source_ids=d['ids'][tune].tolist())
    (ROOT/'results'/f'{args.dataset}-{args.split}{suffix}-indices.json').write_text(json.dumps(split_manifest))
    def save(): out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for j,target in enumerate(d['targets']):
        y=d['y'][:,j]
        configs=[('median',np.zeros((n,1)),lambda _:DummyRegressor(strategy='median'),[None]),
            ('atom_counts_ridge',desc[:,18:],lambda a:make_pipeline(StandardScaler(),Ridge(alpha=a,solver='svd')),[.01,1,100]),
            ('descriptors_ridge',desc,lambda a:make_pipeline(StandardScaler(),Ridge(alpha=a,solver='svd')),[.01,1,100]),
            ('morgan_ridge',fp,lambda a:Ridge(alpha=a,solver='lsqr'),[.1,10,100]),
            ('descriptors_boosting',desc,lambda a:HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=31,l2_regularization=a,early_stopping=False,random_state=SEED),[1,10]),
            ('morgan_descriptors_forest',joined,lambda a:ExtraTreesRegressor(n_estimators=args.forest_trees,max_features='sqrt',min_samples_leaf=a,n_jobs=4,random_state=SEED),[1,5])]
        for name,X,factory,choices in configs:
            if args.skip_forest and name=='morgan_descriptors_forest': continue
            if target=='total_energy_eV' and name not in ('median','atom_counts_ridge','descriptors_ridge'): continue
            start=time.perf_counter(); scores=[]
            for a in choices:
                m=factory(a); m.fit(X[fit],y[fit]); scores.append(mean_absolute_error(y[tune],m.predict(X[tune])))
            a=choices[int(np.argmin(scores))]; tuning_seconds=time.perf_counter()-start
            start=time.perf_counter(); m=factory(a); m.fit(X[tr],y[tr]); fit_seconds=time.perf_counter()-start
            start=time.perf_counter(); pred=m.predict(X[va]); inference_seconds=time.perf_counter()-start
            r=dict(target=str(target),model=name,chosen_hyperparameter=a,inner_maes=[float(x) for x in scores],search_seconds=tuning_seconds,fit_seconds=fit_seconds,inference_seconds=inference_seconds,**metrics(y[va],pred))
            # Bootstrap molecule-level MAE interval is conditional on this split/trained model.
            error=np.abs(pred-y[va]); rng=np.random.default_rng(SEED)
            samples=np.array([error[rng.integers(len(error),size=len(error))].mean() for _ in range(500)])
            r['mae_bootstrap_95']=np.quantile(samples,[.025,.975]).tolist()
            report['results'].append(r); save(); print(args.dataset,args.split,str(target),name,'MAE',round(r['mae'],4),'R2',round(r['r2'],4),'fit_s',round(fit_seconds,2),flush=True)
    # Exact nearest-neighbor comparisons to ALL training molecules for up to 500 val molecules.
    fps=[DataStructs.CreateFromBinaryText(row.tobytes()) for row in np.packbits(d['fp'],axis=1,bitorder='little')]
    trfps=[fps[i] for i in tr]; query=np.random.default_rng(SEED).choice(va,min(500,len(va)),replace=False)
    similarities=[]; nnids=[]
    for i in query:
        sim=np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i],trfps)); similarities.append(float(sim.max())); nnids.append(int(tr[sim.argmax()]))
    report['distribution']['nearest_train_tanimoto']=dict(n_queries=len(query),quantiles=np.quantile(similarities,[.1,.5,.9]).tolist(),fraction_ge_08=float(np.mean(np.asarray(similarities)>=.8)))
    report['nearest_neighbor_subset']=[dict(target=str(t),n_eval=len(query),**metrics(d['y'][query,j],d['y'][nnids,j])) for j,t in enumerate(d['targets'])]
    save(); print('DONE',out,flush=True)
if __name__=='__main__': main()
