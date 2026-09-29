import csv, json, unittest
from prepare import ROOT, digest
class ProtocolTests(unittest.TestCase):
    def test_split_integrity(self):
        manifest=json.loads((ROOT/'data/manifest.json').read_text())
        splits={s:list(csv.DictReader((ROOT/'data'/f'{s}.csv').read_text().splitlines())) for s in ('train','val','test')}
        self.assertEqual(sum(map(len,splits.values())),1128)
        for a,rows in splits.items():
            self.assertEqual(digest(ROOT/'data'/f'{a}.csv'),manifest['files'][f'{a}.csv'])
            self.assertEqual(len(rows),manifest['counts'][a])
            for b,other in splits.items():
                if a==b: continue
                for key in ('id','smiles','scaffold'):
                    self.assertFalse({r[key] for r in rows}&{r[key] for r in other},key)
    def test_padding_does_not_change_prediction(self):
        import torch
        from train import Regressor
        torch.set_num_threads(1)
        for mode in ('plain','string','graph'):
            model=Regressor(mode).eval(); x=torch.tensor([[68,68,80]])
            with torch.no_grad():
                a=model(x,torch.zeros(1,2048)); b=model(torch.nn.functional.pad(x,(0,3)),torch.zeros(1,2048))
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    def test_bad_results_rejected(self):
        from legacy_dashboard import validate
        paths=list((ROOT/'submissions').glob('*.json'))
        if not paths: self.skipTest('No result files yet')
        r=json.loads(paths[0].read_text()); r['best_val_rmse']=float('nan')
        with self.assertRaises(AssertionError): validate(r)
if __name__=='__main__': unittest.main()
