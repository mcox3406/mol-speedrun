"""Build stricter, label-independent chemical groups and audit scaffold leakage."""
import json, time
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from screen import ROOT,SEED,group_split

def generic_topology(mol):
    # Dummy atoms avoid assigning impossible carbon valence to hypervalent S/P.
    graph=Chem.RWMol(); mapping={}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()>1:
            a=Chem.Atom(0); a.SetNoImplicit(True)
            mapping[atom.GetIdx()]=graph.AddAtom(a)
    for bond in mol.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if a in mapping and b in mapping: graph.AddBond(mapping[a],mapping[b],Chem.BondType.SINGLE)
    return Chem.MolToSmiles(graph.GetMol(),isomericSmiles=False)

def main():
    d=np.load(ROOT/'pcqm-200000-features.npz',allow_pickle=False)
    start=time.perf_counter(); generic=[]; hybrid=[]; cyclic=[]; neutral=[]; single=[]; radicals=[]
    for i,s in enumerate(d['smiles']):
        m=Chem.MolFromSmiles(str(s)); sc=MurckoScaffold.GetScaffoldForMol(m)
        ring=bool(sc.GetNumAtoms()); cyclic.append(ring)
        # Keep acyclic molecules in chemically grouped partitions too: full generic topology.
        generic.append(('ring:' if ring else 'chain:')+generic_topology(sc if ring else m))
        hybrid.append('ring:'+Chem.MolToSmiles(sc,isomericSmiles=False) if ring else generic[-1])
        neutral.append(Chem.GetFormalCharge(m)==0);single.append(len(Chem.GetMolFrags(m))==1)
        radicals.append(sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()))
        if i%50000==0: print('chemical groups',i,round(time.perf_counter()-start),flush=True)
    generic=np.asarray(generic);cyclic=np.asarray(cyclic)
    np.savez_compressed(ROOT/'pcqm-chemical-groups.npz',groups=generic,hybrid=np.asarray(hybrid),cyclic=cyclic)
    fps=[DataStructs.CreateFromBinaryText(b.tobytes()) for b in np.packbits(d['fp'],axis=1,bitorder='little')]
    output=dict(n=len(generic),neutral_fraction=float(np.mean(neutral)),single_component_fraction=float(np.mean(single)),radical_fraction=float(np.mean(np.asarray(radicals)>0)),splits={})
    for name,groups in [('typed_scaffold',d['scaffolds']),('generic_scaffold_with_acyclic_topology',generic),('typed_scaffold_with_acyclic_topology',np.asarray(hybrid))]:
        tr,va=group_split(groups,.2,SEED)
        trfps=[fps[i] for i in tr]; query=np.random.default_rng(SEED+3).choice(va,min(1000,len(va)),replace=False)
        maxima=[]
        for q in query: maxima.append(max(DataStructs.BulkTanimotoSimilarity(fps[q],trfps)))
        output['splits'][name]=dict(train=len(tr),val=len(va),unique_groups_train=len(set(groups[tr])),unique_groups_val=len(set(groups[va])),cyclic_fraction_train=float(cyclic[tr].mean()),cyclic_fraction_val=float(cyclic[va].mean()),heavy_atom_quantiles_train=np.quantile(d['desc'][tr,11],[.05,.5,.95]).tolist(),heavy_atom_quantiles_val=np.quantile(d['desc'][va,11],[.05,.5,.95]).tolist(),gap_mean_train=float(d['y'][tr,0].mean()),gap_mean_val=float(d['y'][va,0].mean()),nearest_similarity_quantiles=np.quantile(maxima,[.1,.5,.9]).tolist(),fraction_ge_08=float(np.mean(np.asarray(maxima)>=.8)),fraction_ge_06=float(np.mean(np.asarray(maxima)>=.6)),n_similarity_queries=len(query))
        print(name,output['splits'][name],flush=True)
    (ROOT/'results/chemical-group-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print('audit_seconds',time.perf_counter()-start)
if __name__=='__main__':main()
