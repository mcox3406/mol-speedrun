"""Expand existing chemical folds; reserve every family unseen by the pilot."""
import argparse,csv,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def make_folds(ids,groups,pilot):
 by_id=dict(zip(ids,groups));assignment={}
 for part in ['inner_train','inner_validation','validation']:
  for key in pilot[part]:
   group=int(by_id[key]);previous=assignment.setdefault(group,part)
   assert previous==part,'Conflicting pilot family assignment'
 assert set(pilot['train'])==set(pilot['inner_train'])|set(pilot['inner_validation'])
 return {part:[str(k) for k,g in zip(ids,groups) if assignment.get(int(g),'reserve')==part] for part in ['inner_train','inner_validation','validation','reserve']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);root=a.directory
 audit=json.loads((root/'extraction-audit-0.json').read_text());selected=set(json.loads((root/'extraction-sample-0.json').read_text()));records=list(csv.DictReader((root/'excitation-labels.csv').open()));labels={r['id']:r for r in records};assert len(labels)==len(records),'Repeated source ID';assert selected<=set(labels)|{r['id'] for r in audit['errors']}
 full=np.load(root/'structure-features.npz');all_ids=full['ids'];all_groups=full['groups'];pilot=json.loads(a.pilot.read_text());folds=make_folds(all_ids,all_groups,pilot)
 repeated={first for first,_ in json.loads((root/'structure-audit.json').read_text())['canonical_duplicates']};excluded=[];eligible=[]
 for i,k in enumerate(all_ids):
  reason=None
  if k not in labels:reason='unavailable_or_malformed_excitation_record'
  elif k in repeated:reason='repeated_canonical_structure'
  else:
   r=labels[k];v=np.asarray([float(r[t]) for t in ['S1_eV','T1_eV','f_S1']])
   if not np.isfinite(v).all() or (v[:2]<=0).any() or v[2]<0:reason='nonphysical_excitation_label'
  if reason:excluded.append(dict(id=str(k),reason=reason))
  else:eligible.append(i)
 eligible_ids=set(all_ids[eligible]);folds={k:[v for v in values if v in eligible_ids] for k,values in folds.items()};folds['train']=folds['inner_train']+folds['inner_validation']
 split={k:folds[k] for k in ['train','validation','inner_train','inner_validation']};(a.output/'split.json').write_text(json.dumps(split));(a.output/'reserve-ids.json').write_text(json.dumps(folds['reserve']))
 lookup={k:i for i,k in enumerate(all_ids)};part_ix={k:np.asarray([lookup[v] for v in values],dtype=np.int64) for k,values in folds.items()}
 for field in ['groups','scaffolds','families','smiles']:
  values=full[field]
  for left,right in [('train','validation'),('train','reserve'),('validation','reserve'),('inner_train','inner_validation')]:assert not set(values[part_ix[left]])&set(values[part_ix[right]]),(field,left,right)
 raw_smiles=full['smiles'];export=a.output/'data';export.mkdir(exist_ok=True);manifest=dict(status='full-cohort development calibration; unseen pilot families kept in an untouched reserve, not a certified final test',split_manifest_sha256=digest(a.output/'split.json'),source_archive_md5=json.loads((root/'final_all.hdf5.verified.json').read_text())['md5'],files={})
 for part in ['train','validation']:
  path=export/f'{part}.csv'
  with path.open('w') as f:
   w=csv.DictWriter(f,fieldnames=['id','smiles','S1_eV','T1_eV','delta_ST_eV','f_S1']);w.writeheader()
   for key in split[part]:
    row=dict(labels[key]);row['smiles']=str(raw_smiles[lookup[key]]);w.writerow(row)
  manifest['files'][part]=dict(path=path.name,rows=len(split[part]),sha256=digest(path))
 (export/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 ix=np.concatenate([part_ix['train'],part_ix['validation']]);y=np.asarray([[float(labels[k][t]) for t in ['S1_eV','T1_eV','f_S1']] for k in all_ids[ix]],dtype=np.float32)
 np.savez_compressed(a.output/'baseline.npz',ids=all_ids[ix],fp=full['fp'][ix],desc=full['desc'][ix],y=y)
 old_val=set(pilot['validation']);matched=[k for k in split['validation'] if k in old_val];(a.output/'matched-pilot-validation.json').write_text(json.dumps(matched))
 desc=full['desc'];report=dict(source_metadata_sha256=digest(root/'final_all.csv'),pilot_split_sha256=digest(a.pilot),split_sha256=digest(a.output/'split.json'),script_sha256=digest(Path(__file__)),counts={k:len(v) for k,v in folds.items()},matched_pilot_validation=len(matched),excluded=excluded,exclusion_counts=dict(Counter(r['reason'] for r in excluded)),distribution={})
 for part in ['train','validation','reserve']:
  ix=part_ix[part];sizes=Counter(int(v) for v in all_groups[ix]);report['distribution'][part]=dict(chemical_groups=len(sizes),largest_group=max(sizes.values()),heavy_atom_quantiles=np.quantile(desc[ix,6],[.05,.5,.95]).tolist(),source_counts=dict(Counter(k[:2] for k in folds[part])))
 report['notes']='All original family assignments preserved. Entire previously unseen families are reserved; they are rare/small groups, so this reserve is not distribution-matched and is not yet a frozen final test. No reserve labels enter model training or calibration scores. Repeated canonical structures quarantined independently of error size.'
 (a.output/'cohort-audit.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='excluded'},flush=True)
if __name__=='__main__':main()
