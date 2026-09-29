"""Label-independent candidate domain for the first orbital-gap speedrun."""
import json
import numpy as np
from rdkit import Chem
from screen import ROOT
ALLOWED={1,5,6,7,8,9,14,15,16,17,35,53} # H B C N O F Si P S Cl Br I

def eligibility(mol):
    reasons=[]
    if Chem.GetFormalCharge(mol)!=0: reasons.append('net_charge')
    if len(Chem.GetMolFrags(mol))!=1: reasons.append('multiple_components')
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()): reasons.append('radical_electrons')
    elements={a.GetAtomicNum() for a in mol.GetAtoms()}
    if 6 not in elements: reasons.append('no_carbon')
    if not elements<=ALLOWED: reasons.append('unsupported_elements')
    return reasons

def main():
    d=np.load(ROOT/'pcqm-200000-features.npz',allow_pickle=False)
    excluded={};examples={};mask=[]
    for s in d['smiles']:
        reasons=eligibility(Chem.MolFromSmiles(str(s)));mask.append(not reasons)
        for r in reasons:
            excluded[r]=excluded.get(r,0)+1
            if len(examples.setdefault(r,[]))<3:examples[r].append(str(s))
    np.save(ROOT/'pcqm-eligible.npy',np.asarray(mask))
    report=dict(total=len(mask),eligible=int(sum(mask)),excluded_reasons_overlapping=excluded,examples=examples,allowed_atomic_numbers=sorted(ALLOWED),notes='Operational RDKit criterion; zero assigned radical electrons does not independently establish the DFT spin state. No label, gap, size, conjugation or model-error filtering.')
    (ROOT/'results/domain-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
