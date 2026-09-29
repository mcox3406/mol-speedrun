"""Audit topology concentration and repeated-label disagreement in emitter data."""
import csv, json, argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from audit_groups import generic_topology
RDLogger.DisableLog('rdApp.*')
p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
seen=defaultdict(list);groups=Counter();typed=Counter();invalid=0
for i,r in enumerate(csv.DictReader(a.csv.open())):
    m=Chem.MolFromSmiles(r['smiles'])
    if m is None:invalid+=1;continue
    key=Chem.MolToSmiles(m);seen[key].append([float(r['singlet-triplet value']),float(r['oscillator strength'])])
    if len(seen[key])>1:continue
    sc=Chem.RWMol(MurckoScaffold.GetScaffoldForMol(m));typed[Chem.MolToSmiles(sc,isomericSmiles=False)]+=1
    while True:
        leaves=[v.GetIdx() for v in sc.GetAtoms() if v.GetDegree()==1]
        if not leaves:break
        for ix in reversed(leaves):sc.RemoveAtom(ix)
    groups[generic_topology(sc if sc.GetNumAtoms() else m)]+=1
    if i%50000==0:print('audit',i,flush=True)
ranges=np.array([np.ptp(v,axis=0) for v in seen.values() if len(v)>1]);report=dict(source_rows=i+1,invalid=invalid,unique_canonical=len(seen),duplicate_groups=len(ranges),duplicate_extra_rows=sum(len(v)-1 for v in seen.values()),typed_scaffolds=len(typed),generic_cores=len(groups),largest_generic_cores=groups.most_common(10),duplicate_label_range_quantiles=np.quantile(ranges,[.5,.9,.95,.99,1],axis=0).tolist(),duplicate_groups_delta_ST_range_over_01_eV=int(np.sum(ranges[:,0]>.1)),duplicate_groups_f_range_over_01=int(np.sum(ranges[:,1]>.01)))
a.out.write_text(json.dumps(report,indent=2)+'\n');print(report)
