"""Resume QCDGE archive download, verify publisher MD5, inspect HDF5 schema."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory;root.mkdir(parents=True,exist_ok=True)
status=root/'download-status.json'
def update(state,**extra):
    status.write_text(json.dumps(dict(state=state,updated_unix=time.time(),**extra),indent=2)+'\n'); print(state,extra,flush=True)
try:
    for name,size,md5,url in [('final_all.csv',100269556,'3c14fa79c7b869ded6eae3c654b873f7','https://ndownloader.figshare.com/files/46667578'),('final_all.hdf5',48461817766,'605935abe6bc17efe841dd5b1253b21a','https://ndownloader.figshare.com/files/46667116')]:
        final=root/name;part=root/(name+'.part');target=final if final.exists() else part
        if not final.exists() and (not part.exists() or part.stat().st_size<size):
            update('downloading',file=name,expected_bytes=size)
            subprocess.run(['curl','-L','--fail','--silent','--show-error','--retry','10','--retry-delay','10','--speed-limit','1024','--speed-time','120','-C','-','--output',str(part),url],check=True)
        update('verifying',file=name)
        if target.stat().st_size!=size:raise ValueError(f'{name}: wrong size')
        with target.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
        if actual!=md5:raise ValueError(f'{name}: checksum mismatch; preserving file for inspection')
        if target==part:part.rename(final)
        (root/(name+'.verified.json')).write_text(json.dumps(dict(bytes=size,md5=actual,url=url),indent=2)+'\n')
    update('inspecting_schema')
    import h5py
    records=[]
    with h5py.File(root/'final_all.hdf5','r') as f:
        for key in list(f.keys())[:3]:
            record=dict(id=key,attributes={k:str(v) for k,v in f[key].attrs.items()},datasets={})
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset):record['datasets'][name]=dict(shape=list(obj.shape),dtype=str(obj.dtype),sample=repr(obj[()])[:12000])
            f[key].visititems(visit);records.append(record)
    (root/'schema-samples.json').write_text(json.dumps(records,indent=2)+'\n')
    update('complete',archive=str(root/'final_all.hdf5'),schema=str(root/'schema-samples.json'))
except Exception as e:
    update('failed',error=str(e));raise
