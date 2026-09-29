"""Build a structure-only QCDGE cache before accessing targets."""
import argparse,csv,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from screen import SEED,sha
from domain import eligibility
from audit_groups import generic_topology

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory
 records=list(csv.DictReader((root/'final_all.csv').open()));gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
 ids=[];smiles=[];fps=[];ds=[];scaffolds=[];families=[];excluded=Counter();seen={};duplicates=[];start=time.monotonic()
 for i,r in enumerate(records):
  m=Chem.MolFromSmiles(r['Smiles_rdkit'])
  if m is None:excluded['invalid']+=1;continue
  reasons=eligibility(m)
  if reasons:excluded.update(reasons);continue
  smi=Chem.MolToSmiles(m)
  if smi in seen:duplicates.append([seen[smi],r['Index']]);continue
  key=Chem.MolToInchiKey(m)
  if not key:excluded['inchi_failure']+=1;continue
  seen[smi]=r['Index'];sc=MurckoScaffold.GetScaffoldForMol(m)
  scaffold=('ring:'+Chem.MolToSmiles(sc,isomericSmiles=False)) if sc.GetNumAtoms() else 'chain:'+generic_topology(m)
  counts=Counter(at.GetSymbol() for at in Chem.AddHs(m).GetAtoms())
  ds.append([counts[s] for s in ['H','C','N','O','F','S']]+[m.GetNumHeavyAtoms(),Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m),Descriptors.NumHDonors(m),Descriptors.NumHAcceptors(m),Descriptors.NumRotatableBonds(m),Descriptors.RingCount(m),Descriptors.FractionCSP3(m),Descriptors.BertzCT(m)])
  ids.append(r['Index']);smiles.append(smi);fps.append(gen.GetFingerprintAsNumPy(m));scaffolds.append(scaffold);families.append(key.split('-')[0])
  if i%25000==0:print('features',i,round(time.monotonic()-start),flush=True)
 # Union scaffold groups with Standard InChI connectivity-family groups. This
 # conservatively links stereoisomers and InChI-recognized mobile-H tautomers.
 parent=list(range(len(ids)))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(x,y):parent[find(x)]=find(y)
 for values in [scaffolds,families]:
  first={}
  for i,v in enumerate(values):
   if v in first:union(i,first[v])
   else:first[v]=i
 groups=np.asarray([find(i) for i in range(len(ids))]);sizes=Counter(groups)
 meta=dict(input_rows=len(records),retained=len(ids),excluded_overlapping=dict(excluded),canonical_duplicates=duplicates,rdkit=rdBase.rdkitVersion,metadata_sha256=sha(root/'final_all.csv'),script_sha256=sha(Path(__file__)),seconds=time.monotonic()-start,chemical_groups=len(sizes),largest_groups=[int(v) for _,v in sizes.most_common(10)],grouping='Union of typed nonchiral scaffold (full generic topology for acyclic) and Standard InChI first-block families. No label-dependent filtering. InChI does not cover every possible tautomer transformation.')
 np.savez_compressed(root/'structure-features.npz',ids=np.asarray(ids),smiles=np.asarray(smiles),fp=np.asarray(fps,dtype=np.uint8),desc=np.asarray(ds,dtype=np.float32),scaffolds=np.asarray(scaffolds),families=np.asarray(families),groups=groups)
 (root/'structure-audit.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:v for k,v in meta.items() if k!='canonical_duplicates'},flush=True)
if __name__=='__main__':main()
