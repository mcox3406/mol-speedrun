"""Export the audited development split for model experiments; not a final test."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();root=a.directory;manifest=json.loads(a.manifest.read_text());labels={r['id']:r for r in csv.DictReader((root/'excitation-labels.csv').open())};features=np.load(root/'structure-features.npz');smiles=dict(zip(features['ids'],features['smiles']));out=root/'development';out.mkdir(exist_ok=True);report=dict(status='exploratory; no final test or quality threshold frozen',split_manifest_sha256=digest(a.manifest),source_archive_md5=json.loads((root/'final_all.hdf5.verified.json').read_text())['md5'],files={})
 for split in ['train','validation']:
  path=out/f'{split}.csv'
  with path.open('w') as stream:
   fields=['id','smiles','S1_eV','T1_eV','delta_ST_eV','f_S1'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
   for key in manifest[split]:
    row=dict(labels[key]);row['smiles']=str(smiles[key]);writer.writerow(row)
  report['files'][split]=dict(rows=len(manifest[split]),sha256=digest(path),path=path.name)
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
