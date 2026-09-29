"""Download the compact, frozen v0 data bundle and verify every file."""
import argparse,hashlib,json,shutil,tarfile,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(folder):
 manifest=json.loads((ROOT/'data-manifest.json').read_text())
 if json.loads((folder/'manifest.json').read_text())!=manifest:raise ValueError('Wrong data manifest')
 for name,info in manifest['files'].items():
  if sha(folder/name)!=info['sha256']:raise ValueError(f'Checksum mismatch: {name}')
 return manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/qcdge-v0'));p.add_argument('--archive',type=Path);a=p.parse_args();meta=json.loads((ROOT/'download.json').read_text())
 if a.output.exists():verify(a.output);print(f'Already verified: {a.output}');return
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(dir=a.output.parent) as temp:
  temp=Path(temp);archive=a.archive
  if archive is None:
   archive=temp/'download.tar.gz'
   with urllib.request.urlopen(meta['url']) as response,archive.open('wb') as f:shutil.copyfileobj(response,f)
  if sha(archive)!=meta['sha256']:raise ValueError('Archive checksum mismatch')
  unpack=temp/'data';unpack.mkdir()
  with tarfile.open(archive,'r:gz') as tar:
   expected=set(json.loads((ROOT/'data-manifest.json').read_text())['files'])|{'manifest.json'};members=tar.getmembers()
   if len(members)!=len(expected) or {m.name for m in members}!=expected or any(not m.isfile() for m in members):raise ValueError('Unexpected archive contents')
   for m in members:
    with tar.extractfile(m) as source,(unpack/m.name).open('wb') as f:shutil.copyfileobj(source,f)
  verify(unpack);unpack.rename(a.output)
 print(f'Verified data: {a.output}')
if __name__=='__main__':main()
