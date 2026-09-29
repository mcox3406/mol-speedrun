"""Check excitation-label disagreement among all eligible canonical duplicates."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import h5py,numpy as np
from extract_qcdge import parse_states
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();groups=defaultdict(list)
for first,duplicate in json.loads((a.directory/'structure-audit.json').read_text())['canonical_duplicates']:groups[first].append(duplicate)
rows=[];errors=[]
with h5py.File(a.directory/'final_all.hdf5','r') as f:
 for first,duplicates in groups.items():
  values=[]
  for key in [first,*duplicates]:
   try:values.append(parse_states(f[key]['excited_state']['Info_of_AllExcitedStates'][()][0]))
   except (ValueError,KeyError,TypeError) as e:errors.append(dict(id=key,error=str(e)))
  if len(values)>1:rows.append(dict(ids=[first,*duplicates],label_range=np.ptp(values,axis=0).tolist()))
ranges=np.asarray([r['label_range'] for r in rows]);report=dict(eligible_canonical_duplicate_groups=len(groups),compared_groups=len(rows),errors=errors,targets=['S1_eV','T1_eV','delta_ST_eV','f_S1'],range_quantiles=np.quantile(ranges,[.5,.9,.95,1],axis=0).tolist() if len(rows) else [],groups_with_any_energy_range_over_01_eV=int(np.sum(np.max(ranges[:,:3],axis=1)>.1)) if len(rows) else 0,groups=rows)
a.out.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='groups'})
