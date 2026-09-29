'use strict';
const payload=JSON.parse(document.getElementById('data').textContent);
const entries=payload.entries, runs=entries.length?entries[0].runs:[];
const palette=['#365fb3','#8055aa','#bd5887'];
const seedSelect=document.getElementById('seed');
for(const [i,run] of runs.entries()){
 const option=document.createElement('option');option.value=i;option.textContent=`Seed ${run.seed}`;seedSelect.append(option);
}
function drawLearningCurve(){
 const box=document.getElementById('curve');
 if(!runs.length){box.textContent='No reference measurements available.';return;}
 const selected=seedSelect.value==='all'?runs:[runs[Number(seedSelect.value)]];
 const xmax=Math.max(...runs.map(r=>r.total_seconds))/60,ymin=.8,ymax=Math.ceil(Math.max(...runs.flatMap(r=>r.history.map(h=>Math.max(...Object.entries(payload.protocol.targets).map(([k,v])=>h.metrics[k]/v)))))*2)/2;
 const width=Math.max(280,box.clientWidth-30),right=width-14,left=44;
 const x=v=>left+v/xmax*(right-left),y=v=>244-(Math.min(ymax,Math.max(ymin,v))-ymin)/(ymax-ymin)*207;
 let svg=`<svg viewBox="0 0 ${width} 292" role="img" aria-label="Largest validation error divided by its target, against elapsed minutes. Ratios at or below one pass all five targets."><text x="44" y="18" fill="#626b80" font-size="11">Largest error / target</text>`;
 svg+=`<rect x="44" y="${y(1)}" width="${right-left}" height="${244-y(1)}" fill="#eef1f9"/>`;
 for(let v=1;v<=ymax;v+=.5)svg+=`<path d="M44 ${y(v)}H${right}" stroke="${v===1?'#8055aa':'#e0e4ee'}" stroke-dasharray="${v===1?'4 4':'0'}"/><text x="30" y="${y(v)+4}" text-anchor="end" fill="#626b80" font-size="12">${v}</text>`;
 for(let i=0;i<=4;i++){const v=xmax*i/4;svg+=`<text x="${x(v)}" y="265" text-anchor="middle" fill="#626b80" font-size="12">${v.toFixed(1)}</text>`;}
 svg+=`<text x="${right}" y="287" text-anchor="end" fill="#626b80" font-size="11">Elapsed time (minutes)</text>`;
 for(const run of selected){const points=run.history.map(h=>`${x(h.seconds/60)},${y(Math.max(...Object.entries(payload.protocol.targets).map(([k,v])=>h.metrics[k]/v)))}`).join(' ');svg+=`<polyline points="${points}" fill="none" stroke="${palette[runs.indexOf(run)%3]}" stroke-width="2"/>`;}
 box.innerHTML=svg+'</svg>';
 document.getElementById('curve-info').textContent='Export time is included in the reported total.';
}
document.getElementById('legend').innerHTML=runs.map((r,i)=>`<span><i style="background:${palette[i%3]}"></i>${r.seed} · ${(r.total_seconds/60).toFixed(2)} min</span>`).join('');
if(entries.length){
 const a=entries[0].audit;
 document.getElementById('audit').textContent=`Frozen first-seed audit MAE: S₁ ${a.S1_eV.toFixed(3)} eV; T₁ ${a.T1_eV.toFixed(3)} eV; gap ${a.delta_ST_eV.toFixed(3)} eV; transformed f ${a.log_f.toFixed(3)}; strong-transition f ${a.bright_f.toFixed(3)}. Reported separately; not used for ranking.`;
}
seedSelect.addEventListener('change',drawLearningCurve);drawLearningCurve();
window.addEventListener('resize',drawLearningCurve);
