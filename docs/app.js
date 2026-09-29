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
 const e=entries[0],a=e.audit;document.getElementById('reference-time').textContent=`${(e.median_seconds/60).toFixed(2)} min · median`;
 document.getElementById('audit').textContent=`Frozen first-seed audit MAE: S₁ ${a.S1_eV.toFixed(3)} eV; T₁ ${a.T1_eV.toFixed(3)} eV; gap ${a.delta_ST_eV.toFixed(3)} eV; transformed f ${a.log_f.toFixed(3)}; strong-transition f ${a.bright_f.toFixed(3)}. Reported separately; not used for ranking.`;
}
seedSelect.addEventListener('change',drawLearningCurve);drawLearningCurve();
document.getElementById('copy').addEventListener('click',async function(){try{await navigator.clipboard.writeText(document.getElementById('commands').textContent);this.textContent='Copied';}catch{this.textContent='Select commands to copy';}});

let molecules=[],current=null,viewer=null,mode='3d',webglFailed=false;
const moleculeSelect=document.getElementById('molecule');
const energy=n=>n.toFixed(3);
function drawEnergyLevels(m){
 const zero=248,scale=33,y=e=>zero-e*scale,s=y(m.S1_eV),t=y(m.T1_eV);
 document.getElementById('energy-levels').innerHTML=`<svg viewBox="0 0 470 292" role="img" aria-label="${m.name}: S1 ${energy(m.S1_eV)} electronvolts, T1 ${energy(m.T1_eV)} electronvolts, gap ${energy(m.delta_ST_eV)} electronvolts"><defs><marker id="energy-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="none" stroke="#365fb3"/></marker></defs><text x="22" y="23" fill="#626b80" font-size="12">Energy (eV)</text>${[0,2,4,6].map(v=>`<path d="M54 ${y(v)}H420" stroke="#e9ecf3"/><text x="30" y="${y(v)+4}" text-anchor="end" fill="#626b80" font-size="12">${v}</text>`).join('')}<path d="M88 ${zero}H405" stroke="#69738b" stroke-width="2"/><text x="97" y="${zero+22}" fill="#626b80" font-size="13">S₀ · ground-state reference</text><path d="M126 ${s}H244" stroke="#365fb3" stroke-width="3"/><text x="159" y="${s-13}" fill="#365fb3" font-size="14">S₁ · ${energy(m.S1_eV)}</text><path d="M270 ${t}H391" stroke="#8055aa" stroke-width="3"/><text x="270" y="${t-13}" fill="#8055aa" font-size="14">T₁ · ${energy(m.T1_eV)}</text><path d="M135 ${zero-5}V${s+6}" stroke="#365fb3" stroke-width="1.6" marker-end="url(#energy-arrow)"/><path d="M406 ${s}h9V${t}h-9" fill="none" stroke="#bd5887" stroke-width="1.4"/><text x="424" y="${(s+t)/2+4}" fill="#bd5887" font-size="12">ΔE</text></svg>`;
 document.getElementById('molecule-values').innerHTML=[['S₁',`${energy(m.S1_eV)} eV`],['T₁',`${energy(m.T1_eV)} eV`],['S₁ − T₁',`${energy(m.delta_ST_eV)} eV`],['Oscillator strength f',m.f_S1.toFixed(4)]].map(([k,v])=>`<div><dt>${k}</dt><dd>${v}</dd></div>`).join('');
}
function render3D(){
 if(!current||webglFailed)return;
 try{
  if(!window.$3Dmol)throw new Error('Viewer library unavailable');
  if(!viewer)viewer=$3Dmol.createViewer(document.getElementById('viewer'),{backgroundColor:'#f7f8fc',antialias:true});
  viewer.removeAllModels();viewer.addModel(current.sdf,'sdf');
  for(const [elem,color] of Object.entries({C:'#69738b',H:'#e1e5ef',N:'#476bbe',O:'#c65d7e'}))viewer.setStyle({elem},{stick:{radius:.13,color},sphere:{scale:elem==='H'?.22:.29,color}});
  viewer.resize();viewer.zoomTo();viewer.zoom(1.45);viewer.rotate(45,'x');viewer.rotate(-18,'z');viewer.render();
  document.getElementById('viewer-status').textContent='';
 }catch(error){
  webglFailed=true;document.getElementById('view-3d').disabled=true;setMode('2d');document.getElementById('viewer-status').textContent='3D unavailable in this browser; the 2D structure and labels remain available.';
 }
}
function setMode(next){
 mode=next;document.getElementById('viewer').hidden=mode!=='3d';document.getElementById('structure-2d').hidden=mode!=='2d';
 document.getElementById('view-3d').setAttribute('aria-pressed',String(mode==='3d'));document.getElementById('view-2d').setAttribute('aria-pressed',String(mode==='2d'));
 document.getElementById('view-help').textContent=mode==='3d'?'Drag to rotate · scroll to zoom':'Connectivity drawing · bond lines and atom labels';document.getElementById('reset-view').hidden=mode!=='3d';
 if(mode==='3d')requestAnimationFrame(render3D);
}
function selectMolecule(){
 current=molecules[Number(moleculeSelect.value)];
 document.getElementById('molecule-id').textContent=`${current.formula.replace(/[0-9]/g,d=>'₀₁₂₃₄₅₆₇₈₉'[Number(d)])} · ${current.id}`;
 document.getElementById('smiles').textContent=current.smiles;
 document.getElementById('example-note').textContent=current.note;
 document.getElementById('structure-2d').innerHTML=current.svg;
 document.getElementById('structure-2d').setAttribute('role','img');document.getElementById('structure-2d').setAttribute('aria-label',`${current.name}, two-dimensional connectivity diagram`);
 document.getElementById('viewer').setAttribute('aria-label',`${current.name}, illustrative three-dimensional conformer; drag to rotate`);
 drawEnergyLevels(current);if(mode==='3d')render3D();
}
document.getElementById('view-3d').addEventListener('click',()=>setMode('3d'));document.getElementById('view-2d').addEventListener('click',()=>setMode('2d'));document.getElementById('reset-view').addEventListener('click',render3D);
window.addEventListener('resize',()=>{drawLearningCurve();if(viewer&&mode==='3d'){viewer.resize();viewer.render();}});
fetch('molecules.json').then(r=>{if(!r.ok)throw new Error('Example data unavailable');return r.json();}).then(data=>{
 molecules=data.molecules;
 for(const [i,m] of molecules.entries()){const option=document.createElement('option');option.value=i;option.textContent=m.name;moleculeSelect.append(option);}
 moleculeSelect.addEventListener('change',selectMolecule);selectMolecule();
}).catch(()=>{document.getElementById('viewer-status').textContent='Molecular examples could not be loaded. The benchmark results and protocol remain available.';});
