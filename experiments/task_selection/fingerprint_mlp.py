"""A cheap neural fingerprint control, selected on training-only holdout."""
import argparse, json, time
import numpy as np
import torch
from torch import nn
from screen import ROOT, SEED, group_split, metrics, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-size',type=int,default=50000);p.add_argument('--epochs',type=int,default=25);p.add_argument('--fit-size',type=int,default=0);p.add_argument('--split',choices=['scaffold','generic','chemical'],default='scaffold');p.add_argument('--closed-shell',action='store_true');args=p.parse_args()
    if args.closed_shell or args.split!='scaffold': assert args.train_size==200000
    torch.set_num_threads(4)
    suffix=f'-{args.train_size}' if args.train_size!=50000 else ''
    d=np.load(ROOT/f'pcqm{suffix}-features.npz',allow_pickle=False)
    groups=d['scaffolds'] if args.split=='scaffold' else np.load(ROOT/'pcqm-chemical-groups.npz',allow_pickle=False)['groups' if args.split=='generic' else 'hybrid']
    pool=np.flatnonzero(np.load(ROOT/'pcqm-eligible.npy')) if args.closed_shell else np.arange(len(groups))
    ta,tb=group_split(groups[pool],.2,SEED);tr,va=pool[ta],pool[tb]
    if args.fit_size: tr=np.random.default_rng(SEED+2).permutation(tr)[:args.fit_size]
    ia,ib=group_split(groups[tr],.15,SEED+1);fit=tr[ia];tune=tr[ib]
    raw=np.hstack([d['fp'],d['desc']]).astype(np.float32); y=d['y'][:,0].astype(np.float32)
    def run(indices,eval_ix,epochs):
        torch.manual_seed(SEED);np.random.seed(SEED)
        mean=raw[indices,2048:].mean(0);std=np.maximum(raw[indices,2048:].std(0),1e-6)
        X=torch.from_numpy(raw.copy());X[:,2048:]=(X[:,2048:]-torch.from_numpy(mean))/torch.from_numpy(std)
        mu=y[indices].mean();sd=y[indices].std();Y=torch.from_numpy((y-mu)/sd)
        model=nn.Sequential(nn.Linear(X.shape[1],256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        logs=[];start=time.perf_counter()
        for epoch in range(1,epochs+1):
            model.train()
            for ix in torch.tensor(indices)[torch.randperm(len(indices))].split(512):
                opt.zero_grad();loss=(model(X[ix]).squeeze(1)-Y[ix]).abs().mean();loss.backward();opt.step()
            model.eval()
            with torch.no_grad(): pred=torch.cat([model(X[ix]).squeeze(1)*sd+mu for ix in torch.tensor(eval_ix).split(2048)]).numpy()
            logs.append(dict(epoch=epoch,mae=float(np.abs(pred-y[eval_ix]).mean()),seconds=time.perf_counter()-start))
            if epoch%5==0:print('MLP',args.train_size,'epoch',epoch,logs[-1],flush=True)
        return logs,pred,sum(v.numel() for v in model.parameters())
    search_start=time.perf_counter();inner,_,_=run(fit,tune,args.epochs);best=min(inner,key=lambda r:r['mae'])['epoch'];tune_seconds=time.perf_counter()-search_start
    start=time.perf_counter();curve,pred,params=run(tr,va,best)
    report=dict(closed_shell=args.closed_shell,dataset='pcqm',split=args.split,sample_train_size=args.train_size,fit_size=args.fit_size,counts=dict(train=len(tr),val=len(va)),model='morgan_descriptors_mlp',target='gap_eV',parameters=params,seed=SEED,selected_epochs=best,search_seconds=tune_seconds,fit_and_eval_seconds=time.perf_counter()-start,inner_curve=inner,development_curve=curve,**metrics(y[va],pred),source_sha256=sha(ROOT/'fingerprint_mlp.py'),notes='Select epoch count on training-only holdout, then fresh fit on all sampled train. Outer curve is diagnostic, final epoch is scored. CPU smoke study, feature cache excluded from timing; concurrent processes may affect timings.')
    if args.fit_size: suffix+=f'-fit{args.fit_size}'
    if args.closed_shell: suffix+='-closed-shell'
    out=ROOT/'results'/f'pcqm-{args.split}-mlp{suffix}.json';out.write_text(json.dumps(report,indent=2)+'\n');print(report['mae'],out,flush=True)
if __name__=='__main__':main()
