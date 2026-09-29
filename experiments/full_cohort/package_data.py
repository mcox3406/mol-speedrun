"""Build the v0 release archive from the audited full cohort; no new split selection."""
import argparse,csv,gzip,hashlib,json,shutil,tarfile
from pathlib import Path
import numpy as np

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.source;out=a.output;out.mkdir(parents=True,exist_ok=True)
 for part in ['train','validation']:shutil.copyfile(root/'full-cohort/data'/f'{part}.csv',out/f'{part}.csv')
 shutil.copyfile(root/'full-cohort/split.json',out/'split.json');reserve=json.loads((root/'full-cohort/reserve-ids.json').read_text());wanted=set(reserve);labels={r['id']:r for r in csv.DictReader((root/'excitation-labels.csv').open()) if r['id'] in wanted};features=np.load(root/'structure-features.npz');smiles=dict(zip(features['ids'],features['smiles']))
 with (out/'audit.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['id','smiles','S1_eV','T1_eV','delta_ST_eV','f_S1']);w.writeheader()
  for key in reserve:row=dict(labels[key]);row['smiles']=str(smiles[key]);w.writerow(row)
 manifest=dict(protocol='qcdge-speedrun-v0',source_doi='10.6084/m9.figshare.c.7259125.v1',source_license='CC0-1.0',files={})
 for name,n in [('train.csv',308402),('validation.csv',64549),('audit.csv',31148),('split.json',None)]:
  manifest['files'][name]=dict(sha256=sha(out/name),bytes=(out/name).stat().st_size)
  if n:manifest['files'][name]['rows']=n
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 archive=out.parent/'qcdge-v0.tar.gz'
 with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tar:
  for path in [out/name for name in sorted(set(manifest['files'])|{'manifest.json'})]:
   info=tar.gettarinfo(str(path),arcname=path.name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
   with path.open('rb') as f:tar.addfile(info,f)
 (out.parent/'download.json').write_text(json.dumps(dict(url='https://github.com/mcox3406/mol-speedrun/releases/download/v0.1.0/qcdge-v0.tar.gz',sha256=sha(archive),bytes=archive.stat().st_size),indent=2)+'\n');print(archive,archive.stat().st_size,sha(archive))
if __name__=='__main__':main()
