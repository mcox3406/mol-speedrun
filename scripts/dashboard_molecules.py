"""Prepare training-only teaching examples; geometries are illustrative, not inputs."""
import argparse,csv,json,hashlib
from pathlib import Path
from rdkit import Chem,rdBase
from rdkit.Chem import AllChem,rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
EXAMPLES={'Aa214':('Benzene','An aromatic carbon ring. A low-energy transition can have negligible absorption strength.'),'Aa215':('Pyridine','Replacing a ring carbon with nitrogen changes the electronic states.'),'Aa940':('Aniline','Adding an amino group changes the electronic structure while retaining the benzene scaffold.'),'Aa5357':('Benzaldehyde','A carbonyl group introduces a substantially lower first excitation in this dataset.')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('docs/molecules.json'));a=p.parse_args();found={}
 for r in csv.DictReader(a.train.open()):
  if r['id'] not in EXAMPLES:continue
  mol=Chem.MolFromSmiles(r['smiles']);three=Chem.AddHs(mol);params=AllChem.ETKDGv3();params.randomSeed=20260929
  if AllChem.EmbedMolecule(three,params)!=0:raise ValueError('Conformer generation failed')
  AllChem.MMFFOptimizeMolecule(three)
  drawer=rdMolDraw2D.MolDraw2DSVG(460,300);drawer.drawOptions().clearBackground=False;rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol);drawer.FinishDrawing();svg=drawer.GetDrawingText();svg=svg[svg.index('<svg'):]
  name,note=EXAMPLES[r['id']];found[r['id']]=dict(id=r['id'],name=name,note=note,smiles=r['smiles'],formula=rdMolDescriptors.CalcMolFormula(mol),sdf=Chem.MolToMolBlock(three),svg=svg,**{k:float(r[k]) for k in ['S1_eV','T1_eV','delta_ST_eV','f_S1']})
 if set(found)!=set(EXAMPLES):raise ValueError('Examples missing from training set')
 with a.train.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 a.output.write_text(json.dumps(dict(provenance={'training_csv_sha256':sha,'rdkit':rdBase.rdkitVersion,'geometry':'ETKDGv3, seed 20260929, MMFF; illustrative only, not QCDGE geometries or benchmark inputs','labels':'Frozen QCDGE training labels; these are not model predictions'},molecules=[found[k] for k in EXAMPLES]),indent=2)+'\n')
if __name__=='__main__':main()
