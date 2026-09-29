"""Check label disagreement among canonical duplicate records in the sampled pool."""
import json
import numpy as np
from rdkit import Chem
import screen
screen.PCQM_TRAIN_SIZE=200000
if __name__=='__main__':
    d=screen.features('pcqm'); rows,_,_=screen.read_data('pcqm')
    kept_ids=set(d['ids'].tolist()); labels=dict(zip(d['smiles'],d['y'][:,0])); diffs=[]
    for row in rows:
        if row['id'] in kept_ids: continue
        mol=Chem.MolFromSmiles(row['smiles'])
        if mol is None: continue
        smi=Chem.MolToSmiles(mol,isomericSmiles=True)
        if smi in labels: diffs.append(abs(labels[smi]-row['y'][0]))
    result=dict(duplicate_pairs=len(diffs),absolute_gap_difference_quantiles_eV=np.quantile(diffs,[0,.5,.95,1]).tolist(),pairs_above_01_eV=int(np.sum(np.asarray(diffs)>.1)),pairs_above_1_eV=int(np.sum(np.asarray(diffs)>1)),policy='Study keeps first source-row occurrence of each canonical isomeric SMILES, before making new chemical splits. This audit is diagnostic; it does not filter by targets.')
    (screen.ROOT/'results/duplicate-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
