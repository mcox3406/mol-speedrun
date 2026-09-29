"""Build a static v0 competition dashboard from recorded results; no server/database."""
import html,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 protocol=json.loads((ROOT/'protocol.json').read_text());targets=protocol['targets'];index=json.loads((ROOT/'records/index.json').read_text());entries=[]
 for entry in index:
  summary=json.loads((ROOT/entry['summary']).read_text());runs=[json.loads((ROOT/p).read_text()) for p in entry['reports']]
  if sorted(r['seed'] for r in runs)!=protocol['seeds'] or any(r['status']!='qualified' for r in runs):raise ValueError('Incomplete record panel')
  if any(r['protocol_sha256']!=__import__('hashlib').sha256((ROOT/'protocol.json').read_bytes()).hexdigest() for r in runs):raise ValueError('Stale record protocol')
  entries.append(dict(name=entry['name'],kind=entry['kind'],median_seconds=summary['median_seconds'],runs=runs,audit=summary['audit'],commit=summary['commit']))
 rows=[];cpu=json.loads((ROOT/'experiments/full_cohort/results/cpu-24310981.json').read_text());gpu=json.loads((ROOT/'experiments/full_cohort/results/plain-24310980.json').read_text())
 for model,label in [('atom_counts_ridge','Atom counts + ridge'),('morgan_ridge','Morgan fingerprint + ridge'),('morgan_descriptors_forest','Fingerprint forest'),('morgan_descriptors_mlp','Fingerprint MLP · 3 seeds')]:
  rr=[r for r in cpu['results'] if r['model']==model];values=[statistics.mean(r['validation'][k]['mae'] for r in rr) for k in ['S1_eV','T1_eV','delta_ST_eV','log10_1_plus_f_over_0001']];rows.append('<tr><td>'+label+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>')
 values=[gpu['validation'][k]['mae'] for k in ['S1_eV','T1_eV','delta_ST_eV','log10_1_plus_f_over_0001']];rows.append('<tr><td>SMILES transformer · calibration</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>')
 podium=''.join('<tr><td>'+html.escape(e['name'])+'</td><td>'+f"{e['median_seconds']/60:.2f} min"+'</td><td>3 / 3</td><td>'+html.escape(e['kind'])+'</td></tr>' for e in sorted(entries,key=lambda x:x['median_seconds'])) or '<tr><td colspan="4">Reference runs are being verified. No accepted records yet.</td></tr>'
 gates=''.join(f'<tr><td class="target-name">{label}</td><td>≤ {targets[key]:.3f}<small>{unit}</small></td></tr>' for key,label,unit in [('S1_eV','S₁ excitation energy','eV'),('T1_eV','T₁ excitation energy','eV'),('delta_ST_eV','S₁ − T₁ gap','eV'),('log_f','log₁₀(1 + f / 0.001)',''),('bright_f','Raw f, where true f ≥ 0.1','')])
 page=HTML.replace('GATES',gates).replace('LEADER_ROWS',podium).replace('BASELINE_ROWS',''.join(rows)).replace('PAYLOAD',json.dumps(dict(protocol=protocol,entries=entries),allow_nan=False).replace('<','\\u003c'))
 (ROOT/'docs').mkdir(exist_ok=True);(ROOT/'docs/index.html').write_text(page);(ROOT/'docs/.nojekyll').touch();print(f'Built v0 dashboard: {len(entries)} reference/record panels')
HTML=(ROOT/'web/index.html').read_text()
if __name__=='__main__':main()
