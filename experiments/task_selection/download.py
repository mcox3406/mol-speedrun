"""Download exact public sources used in the study, then verify checksums."""
import hashlib, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    raw=ROOT/'raw';raw.mkdir(exist_ok=True)
    for name,source in json.loads((ROOT/'sources.json').read_text()).items():
        path=raw/name
        if not path.exists():
            urllib.request.urlretrieve(source['url'],path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'],name
        print('Verified',name)
