"""Pinned ESOL input; structure-grouped, deterministic scaffold holdout."""
import csv, hashlib, io, json, urllib.request
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT = Path(__file__).resolve().parent
URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
SHA = '8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    raw = ROOT / 'esol-source.csv'
    if not raw.exists():
        raw.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    assert digest(raw) == SHA, 'Upstream dataset changed'
    groups = defaultdict(list)
    for i, row in enumerate(csv.DictReader(io.StringIO(raw.read_text()))):
        mol = Chem.MolFromSmiles(row['smiles'].strip())
        assert mol is not None, i
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        # All acyclic molecules stay together; avoids claiming empty scaffolds are disjoint.
        groups[scaffold].append(dict(id=i, smiles=smi, scaffold=scaffold,
            y=float(row['measured log solubility in mols per litre'])))
    splits = {s: [] for s in ('train', 'val', 'test')}
    for key, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        dest = 'train' if len(splits['train']) + len(rows) <= 902 else 'val' if len(splits['val']) + len(rows) <= 113 else 'test'
        splits[dest].extend(rows)
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    for name, rows in splits.items():
        with (out / f'{name}.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=['id', 'smiles', 'scaffold', 'y']); w.writeheader(); w.writerows(sorted(rows, key=lambda r:r['id']))
    manifest = dict(protocol='esol-scaffold-v0', source=URL, source_sha256=SHA,
        rdkit=rdBase.rdkitVersion, target='measured log10 solubility (mol/L)',
        policy='Canonical isomeric SMILES; no salt stripping; duplicates retained within scaffold groups; empty scaffold grouped; largest groups first, lexical ties; 902/113 train/val caps.',
        counts={k:len(v) for k,v in splits.items()}, files={f'{s}.csv':digest(out / f'{s}.csv') for s in splits})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__': main()
