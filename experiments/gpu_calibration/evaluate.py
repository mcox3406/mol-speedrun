"""Reevaluate a fixed checkpoint, preserving original labels for brightness bins."""
import argparse,csv,json,os
from pathlib import Path
import torch
from calibrate import Regressor,four,digest,intensity_metrics
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');r=json.loads(a.run.read_text());manifest=json.loads((a.data/'manifest.json').read_text());assert digest(a.data/'manifest.json')==r['manifest_sha256'];path=a.data/'validation.csv';assert digest(path)==manifest['files']['validation']['sha256'];rows=list(csv.DictReader(path.open()));assert len(rows)==r['counts']['validation']
x=torch.zeros(len(rows),r['max_sequence_length'],dtype=torch.long,device='cuda')
for i,row in enumerate(rows):
 code=[ord(c)+1 for c in row['smiles']];x[i,:len(code)]=torch.tensor(code,device='cuda')
y=torch.tensor([[float(row['S1_eV']),float(row['T1_eV']),float(row['f_S1'])] for row in rows],device='cuda');raw_f=y[:,2].clone();y[:,2]=torch.log10(1+y[:,2]/.001)
checkpoint=torch.load(a.checkpoint,map_location='cuda',weights_only=True);model=Regressor(checkpoint['width'],checkpoint['layers'],checkpoint['heads']).cuda().eval();model.load_state_dict(checkpoint['model']);parts=[]
with torch.no_grad():
 for batch in x.split(r['configuration']['batch_size']):
  with torch.autocast('cuda',dtype=torch.bfloat16):p=model(batch)
  parts.append(p.float()*checkpoint['std']+checkpoint['mean'])
pred=torch.cat(parts);truth=four(y);prediction=four(pred);error=(prediction-truth).abs().mean(0);mse=(prediction-truth).square().mean(0);variance=(truth-truth.mean(0)).square().mean(0);names=['S1_eV','T1_eV','delta_ST_eV','log10_1_plus_f_over_0001'];scores={name:dict(mae=float(error[j]),rmse=float(mse[j].sqrt()),r2=float(1-mse[j]/variance[j])) for j,name in enumerate(names)};scores['raw_f']=intensity_metrics(pred[:,2],raw_f)
report=dict(status='completed',original_run_sha256=digest(a.run),checkpoint_sha256=digest(a.checkpoint),evaluator_sha256=digest(Path(__file__)),model_module_sha256=digest(Path(__file__).with_name('calibrate.py')),slurm_job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),validation=scores,notes='Same fixed checkpoint, no retraining or selection. Corrects brightness-bin membership by using original raw labels rather than a float32 log/inverse-log roundtrip.')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
