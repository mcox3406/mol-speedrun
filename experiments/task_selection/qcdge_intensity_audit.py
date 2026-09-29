"""Inspect oscillator-strength ambiguity from rounded degenerate S1 energies."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import h5py,numpy as np

def init(path):
 global archive
 archive=h5py.File(path,'r')
def audit(key):
 states=json.loads(archive[key]['excited_state']['Info_of_AllExcitedStates'][()][0]);s=[(float(v['excitation_e_eV'].split()[0]),float(v['oscillator_trength'])) for v in states.values() if v['state_type']=='Singlet'];energy=min(v[0] for v in s);f=[v[1] for v in s if v[0]==energy]
 return dict(id=key,tied_states=len(f),f_min=min(f),f_max=max(f),f_sum=sum(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();manifest=json.loads((a.results/'qcdge-100000-indices.json').read_text());ids=np.random.default_rng(20260929).choice(manifest['train']+manifest['validation'],4000,replace=False).tolist()
 with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(str(a.directory/'final_all.hdf5'),)) as pool:rows=list(pool.map(audit,ids,chunksize=100))
 ties=[r for r in rows if r['tied_states']>1];report=dict(sample_size=len(rows),ties_at_published_energy_precision=len(ties),ties_with_f_range_over_0001=sum(r['f_max']-r['f_min']>.001 for r in ties),examples=ties[:20],seed=20260929)
 (a.results/'qcdge-intensity-label-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
