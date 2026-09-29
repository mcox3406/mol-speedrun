'use strict';
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
window.addEventListener('resize',()=>{if(viewer&&mode==='3d'){viewer.resize();viewer.render();}});
fetch('molecules.json').then(r=>{if(!r.ok)throw new Error('Example data unavailable');return r.json();}).then(data=>{
 molecules=data.molecules;
 for(const [i,m] of molecules.entries()){const option=document.createElement('option');option.value=i;option.textContent=m.name;moleculeSelect.append(option);}
 moleculeSelect.addEventListener('change',selectMolecule);selectMolecule();
}).catch(()=>{document.getElementById('viewer-status').textContent='Molecular examples could not be loaded. The benchmark results and protocol remain available.';});
