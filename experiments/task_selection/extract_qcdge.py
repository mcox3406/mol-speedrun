"""Extract spin-resolved excitation labels; no assumption about state ordering."""
import argparse,csv,json,math,time
from pathlib import Path
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def parse_states(raw):
    states=json.loads(raw); by_spin={'Singlet':[],'Triplet':[]}
    for state in states.values():
        value,unit=state['excitation_e_eV'].split()
        if unit!='eV':raise ValueError('Unexpected excitation unit')
        energy=float(value); strength=float(state['oscillator_trength'])
        if not math.isfinite(energy) or not math.isfinite(strength):raise ValueError('Nonfinite label')
        by_spin[state['state_type']].append((energy,strength))
    if any(len(v)!=10 for v in by_spin.values()):raise ValueError('Expected ten states per spin')
    s,f=min(by_spin['Singlet']);t,_=min(by_spin['Triplet'])
    return s,t,s-t,f

def init_worker(path):
    global archive
    archive=h5py.File(path,'r')

def read_batch(keys):
    result=[]
    for key in keys:
        try:result.append((key,parse_states(archive[key]['excited_state']['Info_of_AllExcitedStates'][()][0]),None))
        except (KeyError,ValueError,TypeError) as e:result.append((key,None,str(e)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--sample',type=int,default=100000);p.add_argument('--workers',type=int,default=4);a=p.parse_args();root=a.directory
    metadata={r['Index']:r for r in csv.DictReader((root/'final_all.csv').open())};out=root/'excitation-labels.csv';done=set()
    if out.exists():done={r['id'] for r in csv.DictReader(out.open())}
    keys=sorted(metadata)
    if a.sample:keys=sorted(np.random.default_rng(20260929).choice(keys,min(a.sample,len(keys)),replace=False).tolist())
    (root/f'extraction-sample-{a.sample}.json').write_text(json.dumps(keys))
    remaining=[k for k in keys if k not in done]; batches=[remaining[i:i+256] for i in range(0,len(remaining),256)]
    errors=[];start=time.monotonic();count=0
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(str(root/'final_all.hdf5'),)) as pool, out.open('a') as stream:
        writer=csv.writer(stream)
        if not done:writer.writerow(['id','smiles','S1_eV','T1_eV','delta_ST_eV','f_S1'])
        for batch_index,batch in enumerate(pool.map(read_batch,batches),1):
            for key,labels,error in batch:
                if error:errors.append(dict(id=key,error=error));continue
                writer.writerow([key,metadata[key]['Smiles_rdkit'],*labels]);count+=1
            stream.flush()
            if batch_index%10==0:print(json.dumps(dict(extracted=count,already_present=len(done),seconds=round(time.monotonic()-start,1),errors=len(errors))),flush=True)
    report=dict(sample_requested=a.sample,extracted_this_run=count,already_present=len(done),metadata_rows=len(metadata),errors=errors,seconds=time.monotonic()-start)
    (root/f'extraction-audit-{a.sample}.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':main()
