"""Recompute v0 scores from predictions. A passing check is not record acceptance."""
import argparse,json,math,re,statistics
from pathlib import Path
from evaluate import ROOT,PROTOCOL,digest,score_files,qualifies

def require(condition,message):
 if not condition:raise ValueError(message)
def local_artifact(report,name):
 require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'Artifact must be a filename beside the report')
 return report.parent/name

def validate_reports(paths,data,audit_path):
 manifest=json.loads((ROOT/'data-manifest.json').read_text())
 for part in ['validation.csv','audit.csv']:require(digest(data/part)==manifest['files'][part]['sha256'],'Wrong evaluation data')
 require(len(paths)==len(PROTOCOL['seeds']),'Exactly three runs required');runs=[];sources=[];commits=[]
 for path in paths:
  r=json.loads(path.read_text());require(r['protocol']==PROTOCOL['id'],'Wrong protocol');require(r['protocol_sha256']==digest(ROOT/'protocol.json'),'Protocol changed');require(r['manifest_sha256']==digest(ROOT/'data-manifest.json'),'Data contract changed');require(type(r['seed']) is int,'Invalid seed');require(r['status']=='qualified','Every run must qualify; smoke/failed runs cannot rank')
  require(r['gpu']==PROTOCOL['limits']['gpu'] and r['gpu_count']==1 and r['cpu_threads']<=4 and r['cpu_threads']>0,'Wrong reference hardware allocation')
  elapsed=r['total_seconds'];require(type(elapsed) in (float,int) and math.isfinite(elapsed) and 0<elapsed<=PROTOCOL['limits']['seconds'],'Invalid elapsed time');require(0<=r['preprocessing_seconds']<=elapsed,'Invalid preprocessing time')
  previous=0.;require(bool(r['history']),'Missing evaluation history')
  for h in r['history']:
   require(math.isfinite(h['seconds']) and previous<=h['seconds']<=elapsed,'Invalid evaluation clock');previous=h['seconds']
   for k in PROTOCOL['targets']:require(math.isfinite(h['metrics'][k]) and h['metrics'][k]>=0,'Invalid history score')
  pred=local_artifact(path,r['predictions_file']);require(digest(pred)==r['predictions_sha256'],'Prediction checksum mismatch');scores=score_files(data/'validation.csv',pred);require(qualifies(scores),'Predictions fail quality targets')
  for k,v in scores.items():
   require(math.isclose(v,r['metrics'][k],abs_tol=1e-9,rel_tol=1e-7),'Reported score differs from predictions')
   require(math.isclose(v,r['history'][-1]['metrics'][k],abs_tol=1e-9,rel_tol=1e-7),'Final history differs from predictions')
  require(re.fullmatch(r'[0-9a-f]{7,40}',r['commit']) is not None,'Missing source revision');require(bool(r['source_sha256']) and all(re.fullmatch(r'[0-9a-f]{64}',v) for v in r['source_sha256'].values()),'Invalid source hashes');require(re.fullmatch(r'[0-9a-f]{64}',r['checkpoint_sha256']) is not None,'Missing checkpoint hash')
  checkpoint=local_artifact(path,r['checkpoint_file'])
  if checkpoint.exists():require(digest(checkpoint)==r['checkpoint_sha256'],'Checkpoint checksum mismatch')
  sources.append(r['source_sha256']);commits.append(r['commit']);runs.append(dict(seed=r['seed'],seconds=elapsed,metrics=scores,report_sha256=digest(path),checkpoint_sha256=r['checkpoint_sha256']))
 require(sorted(r['seed'] for r in runs)==PROTOCOL['seeds'],'Missing, repeated, or substituted seed');require(all(v==sources[0] for v in sources),'Code differs between seeds');require(len(set(commits))==1,'Revisions differ between seeds')
 audit=json.loads(audit_path.read_text());first=next(r for r in runs if r['seed']==PROTOCOL['seeds'][0]);require(audit['protocol']==PROTOCOL['id'] and audit['seed']==first['seed'],'Wrong audit seed/protocol');require(audit['run_sha256']==first['report_sha256'] and audit['checkpoint_sha256']==first['checkpoint_sha256'],'Audit not tied to frozen first-seed run');pred=audit_path.with_suffix('.predictions.csv');require(digest(pred)==audit['predictions_sha256'],'Audit prediction checksum mismatch');scores=score_files(data/'audit.csv',pred)
 for k,v in scores.items():require(math.isclose(v,audit['metrics'][k],abs_tol=1e-9,rel_tol=1e-7),'Audit score differs from predictions')
 return dict(protocol=PROTOCOL['id'],status='checks_passed_pending_reproduction',median_seconds=statistics.median(r['seconds'] for r in runs),runs=sorted(runs,key=lambda r:r['seed']),audit=scores,commit=commits[0],notes='Automatic checks cannot establish honest timing, hardware, train-only fitting, or provenance. Maintainer reproduction required.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/qcdge-v0'));p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=validate_reports(a.runs,a.data,a.audit);text=json.dumps(r,indent=2,allow_nan=False)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
 print(text)
if __name__=='__main__':main()
