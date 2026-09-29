import unittest
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from audit_groups import generic_topology
from screen import group_split, SEED
from domain import eligibility
class SelectionTests(unittest.TestCase):
    def test_domain_excludes_radicals_but_retains_neutral_resonance_forms(self):
        self.assertIn('radical_electrons',eligibility(Chem.MolFromSmiles('[CH3]')))
        self.assertEqual(eligibility(Chem.MolFromSmiles('C[N+](=O)[O-]')),[])
        self.assertIn('multiple_components',eligibility(Chem.MolFromSmiles('CC.[Na+]')))
    def test_topology_groups_ring_heteroatom_variants(self):
        self.assertEqual(generic_topology(Chem.MolFromSmiles('c1ccccc1')),generic_topology(Chem.MolFromSmiles('c1ccncc1')))
        self.assertNotEqual(generic_topology(Chem.MolFromSmiles('CCCC')),generic_topology(Chem.MolFromSmiles('CC(C)C')))
        self.assertEqual(generic_topology(Chem.MolFromSmiles('CCO')),generic_topology(Chem.MolFromSmiles('OCC')))
    def test_hypervalent_atoms_have_valid_topology(self):
        value=generic_topology(Chem.MolFromSmiles('CS(=O)(=O)C'))
        self.assertIsNotNone(Chem.MolFromSmiles(value))
    def test_fingerprint_packing_preserves_similarity(self):
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        for s in ['c1ccccc1','CCO','CS(=O)(=O)C']:
            mol=Chem.MolFromSmiles(s); fp=gen.GetFingerprint(mol)
            packed=DataStructs.CreateFromBinaryText(np.packbits(gen.GetFingerprintAsNumPy(mol),bitorder='little').tobytes())
            self.assertEqual(DataStructs.TanimotoSimilarity(fp,packed),1.0)
    def test_chemical_groups_do_not_cross_either_holdout(self):
        groups=np.repeat(np.arange(40),5)
        train,val=group_split(groups,.2,SEED)
        a,b=group_split(groups[train],.15,SEED+1)
        self.assertFalse(set(groups[train])&set(groups[val]))
        self.assertFalse(set(groups[train[a]])&set(groups[train[b]]))
        small=np.random.default_rng(SEED+2).permutation(train)[:30]
        self.assertTrue(set(small)<=set(train))
        self.assertFalse(set(groups[small])&set(groups[val]))
if __name__=='__main__': unittest.main()
