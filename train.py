"""Small supervised SMILES encoder and two intentionally simple memory ablations."""
import argparse, csv, hashlib, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from prepare import ROOT, digest

class Regressor(nn.Module):
    def __init__(self, mode):
        super().__init__(); self.mode = mode
        self.token = nn.Embedding(129, 64, padding_idx=0)
        self.pos = nn.Embedding(512, 64)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, .1, batch_first=True), 2, enable_nested_tensor=False)
        if mode == 'string': self.memory = nn.Embedding(2048, 64); self.gate = nn.Linear(64, 64)
        if mode == 'graph': self.memory = nn.Linear(2048, 64, bias=False); self.gate = nn.Linear(64, 64)
        self.head = nn.Linear(64, 1)
    def forward(self, x, fp):
        mask = x.ne(0); h = self.token(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        if self.mode == 'string':
            previous = torch.nn.functional.pad(x[:, :-1], (1,0))
            h = h + torch.sigmoid(self.gate(h)) * self.memory((previous * 131 + x) % 2048)
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if self.mode == 'graph': h = h + torch.sigmoid(self.gate(h)) * self.memory(fp)
        return self.head(h).squeeze(-1)

def load():
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    for name, sha in manifest['files'].items():
        assert digest(ROOT/'data'/name) == sha, f'Data drift: {name}'
    rows = {s:list(csv.DictReader((ROOT/'data'/f'{s}.csv').open())) for s in ('train','val')}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = {}
    for s, rr in rows.items():
        encoded = [[ord(c)+1 for c in r['smiles']] for r in rr]
        assert all(max(e)<129 and len(e)<=512 for e in encoded)
        x = torch.zeros(len(rr), max(map(len, encoded)), dtype=torch.long)
        for i,e in enumerate(encoded): x[i,:len(e)] = torch.tensor(e)
        fp = torch.tensor(np.stack([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(r['smiles'])) for r in rr]), dtype=torch.float32)
        y = torch.tensor([float(r['y']) for r in rr])
        result[s] = (x,fp,y)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model', choices=['mean','ridge','plain','string','graph'], default='ridge')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--threads', type=int, default=4); p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); assert args.epochs>0 and args.threads>0
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    start=time.perf_counter(); data=load(); prep=time.perf_counter()-start
    x,fp,y=data['train']; vx,vfp,vy=data['val']; mu=y.mean(); sd=y.std()
    history=[]; params=0
    if args.model in ('mean','ridge'):
        if args.model=='mean': pred=np.full(len(vy),mu.item())
        else:
            model=Ridge(alpha=10); model.fit(fp.numpy(),y.numpy()); pred=model.predict(vfp.numpy()); params=2049
        history.append(dict(epoch=1,seconds=time.perf_counter()-start,rmse=float(np.sqrt(np.mean((pred-vy.numpy())**2)))))
    else:
        model=Regressor(args.model); params=sum(t.numel() for t in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        for epoch in range(1,args.epochs+1):
            model.train()
            for ix in torch.randperm(len(y)).split(64):
                bx=x[ix]; bx=bx[:,:int(bx.ne(0).sum(1).max())]
                loss=((model(bx,fp[ix])-(y[ix]-mu)/sd)**2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            model.eval()
            with torch.no_grad():
                pred=torch.cat([model(vx[ix],vfp[ix])*sd+mu for ix in torch.arange(len(vy)).split(64)])
                rmse=float(((pred-vy)**2).mean().sqrt())
            history.append(dict(epoch=epoch,seconds=time.perf_counter()-start,rmse=rmse))
            print(args.model,args.seed,epoch,round(rmse,4),flush=True)
    best=min(history,key=lambda r:r['rmse'])
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip())
    result=dict(protocol='esol-scaffold-v0',model=args.model,seed=args.seed,epochs=args.epochs,
        threads=args.threads,parameters=params,device='cpu',hardware=platform.machine()+' / '+platform.processor(),
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        git_commit=revision,dirty=dirty,manifest_sha256=digest(ROOT/'data/manifest.json'),
        train_source_sha256=digest(Path(__file__)),preprocessing_seconds=prep,total_seconds=time.perf_counter()-start,
        best_val_rmse=best['rmse'],best_epoch=best['epoch'],history=history,status='exploratory')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(args.output)
if __name__=='__main__': main()
