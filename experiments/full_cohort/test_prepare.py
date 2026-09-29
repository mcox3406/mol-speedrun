import unittest
from prepare import make_folds
class FoldExpansionTests(unittest.TestCase):
 def setUp(self):
  self.ids=['a','b','c','d','e','f'];self.groups=[0,0,1,2,2,3];self.pilot={'train':['a','c'],'inner_train':['a'],'inner_validation':['c'],'validation':['d']}
 def test_expand_existing_families_and_reserve_unseen(self):
  self.assertEqual(make_folds(self.ids,self.groups,self.pilot),{'inner_train':['a','b'],'inner_validation':['c'],'validation':['d','e'],'reserve':['f']})
 def test_reject_conflicting_family_assignments(self):
  with self.assertRaises(AssertionError):make_folds(self.ids,self.groups,{**self.pilot,'validation':['b']})
 def test_reject_inconsistent_inner_partition(self):
  with self.assertRaises(AssertionError):make_folds(self.ids,self.groups,{**self.pilot,'train':['a']})
if __name__=='__main__':unittest.main()
