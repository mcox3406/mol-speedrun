"""Validate result files and build a dependency-free, static dashboard."""
import html, json, math
from pathlib import Path
from prepare import ROOT, digest

def validate(r):
    assert r['protocol']=='esol-scaffold-v0'
    assert r['manifest_sha256']==digest(ROOT/'data/manifest.json'), 'Wrong dataset manifest'
    assert r['model'] in ('mean','ridge','plain','string','graph')
    assert isinstance(r['seed'],int) and not isinstance(r['seed'],bool)
    assert r['status']=='exploratory'
    assert isinstance(r['git_commit'],str) and len(r['git_commit'])==40
    assert r['history'] and r['device']=='cpu'
    for k in ('total_seconds','preprocessing_seconds','best_val_rmse'):
        assert math.isfinite(r[k]) and r[k]>=0
    previous=0
    for h in r['history']:
        assert math.isfinite(h['rmse']) and h['rmse']>=0
        assert math.isfinite(h['seconds']) and previous<=h['seconds']<=r['total_seconds']
        previous=h['seconds']
    assert math.isclose(r['best_val_rmse'],min(h['rmse'] for h in r['history']),abs_tol=1e-8)

def main():
    results=[]
    for path in sorted((ROOT/'submissions').glob('*.json')):
        r=json.loads(path.read_text()); validate(r); r['file']=path.name; results.append(r)
    rows=[]
    for r in sorted(results,key=lambda r:r['best_val_rmse']):
        values=[r['model'],str(r['seed']),f"{r['best_val_rmse']:.3f}",f"{r['total_seconds']:.2f}",f"{r['parameters']:,}",r['hardware'],r['git_commit'][:8]+(' (dirty)' if r['dirty'] else '')]
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Mol Speedrun · ESOL pilot</title>
<style>body{font:17px/1.6 system-ui,sans-serif;background:#f4f3ee;color:#173c3b;max-width:1100px;margin:60px auto;padding:0 24px}h1{font-size:clamp(36px,7vw,70px);line-height:1.1;margin:20px 0}h2{margin-top:40px}a{color:#006c64}.eyebrow{letter-spacing:.12em;text-transform:uppercase;font-size:13px}.cards{display:flex;flex-wrap:wrap;gap:18px}.cards p{background:white;padding:20px;flex:1;border-top:3px solid #00877b}table{border-collapse:collapse;width:100%;font-size:14px;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd}.scroll{overflow:auto}button,select{font:inherit;padding:7px}svg{width:100%;background:white}small{color:#49605f}</style>
<p class="eyebrow">Molecular property prediction / protocol v0</p><h1>How quickly can we<br>learn solubility?</h1>
<p>A small laboratory for SMILES transformers, string memory, and graph features. Every point below comes from a recorded local run.</p>
<div class="cards"><p><strong>ESOL</strong><br>Measured log₁₀ mol/L</p><p><strong>COUNTS</strong><br>Train / validation / test</p><p><strong>Exploratory</strong><br>No official speed records yet</p></div>
<p>Lower validation RMSE is better. Times include feature construction, model initialization, training and validation, but exclude imports and download. Compare speed only on matching hardware, threads and software. The test set has not been evaluated.</p>
<h2>Validation learning curves</h2><label>Run <select id="run"></select></label><div id="plot"></div><p id="curve" aria-live="polite"></p>
<h2>Recorded runs</h2><p>Sorted by best validation RMSE; this is an exploratory run table, not a statistically qualified leaderboard.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Seed</th><th>RMSE</th><th>Total seconds</th><th>Parameters</th><th>Hardware</th><th>Revision</th></tr></thead><tbody>ROWS</tbody></table></div>
<h2>Submit an experiment</h2><p>Run the trainer, add its JSON to <code>submissions/</code>, and open a pull request with the code revision, exact command, environment and all seeds. CI checks the result format; maintainers reproduce claims. No login service or database.</p>
<p><a href="https://github.com/mcox3406/mol-speedrun">Repository, commands and rules</a></p>
<small>Single scaffold split; seed variability does not measure uncertainty across chemical space. Graph means pooled Morgan-feature fusion, not a full graph Engram implementation.</small>
<script id="results" type="application/json">DATA</script><script>
const runs=JSON.parse(document.getElementById('results').textContent), select=document.getElementById('run');
for(const [i,r] of runs.entries()){const o=document.createElement('option');o.value=i;o.textContent=`${r.model} · seed ${r.seed} · ${r.file}`;select.append(o)}
function draw(){if(!runs.length){document.getElementById('plot').textContent='No runs yet.';return}const r=runs[select.value||0], h=r.history, xmax=Math.max(...h.map(v=>v.seconds),.01), ymax=Math.max(...h.map(v=>v.rmse),.01)*1.1;const points=h.map(v=>`${60+v.seconds/xmax*700},${270-v.rmse/ymax*230}`).join(' ');document.getElementById('plot').innerHTML=`<svg viewBox="0 0 800 320" role="img" aria-label="Validation RMSE versus elapsed seconds"><path d="M60 30V270H770" fill="none" stroke="#999"/><polyline points="${points}" fill="none" stroke="#00877b" stroke-width="3"/>${h.map(v=>`<circle cx="${60+v.seconds/xmax*700}" cy="${270-v.rmse/ymax*230}" r="4" fill="#00877b"/>`).join('')}<text x="60" y="300">0</text><text x="620" y="300">${xmax.toFixed(1)} seconds</text><text x="10" y="35">${ymax.toFixed(1)}</text><text x="10" y="270">0</text></svg>`;document.getElementById('curve').textContent=h.map(v=>`Epoch ${v.epoch}: RMSE ${v.rmse.toFixed(3)} at ${v.seconds.toFixed(2)}s`).join(' • ')}select.onchange=draw;draw();
</script></html>'''
    page=page.replace('COUNTS',' / '.join(str(manifest['counts'][s]) for s in ('train','val','test'))).replace('ROWS',''.join(rows)).replace('DATA',json.dumps(results,allow_nan=False).replace('<','\\u003c'))
    out=ROOT/'docs';out.mkdir(exist_ok=True);(out/'index.html').write_text(page)
    print(f'Validated {len(results)} runs; wrote docs/index.html')
if __name__=='__main__': main()
