import unittest
from scripts.prepare_training import split_records
from scripts.train import assert_splits
from search_r1_team.config import Config

class TestTrainingData(unittest.TestCase):
    def rows(self):
        return [{'id':str(i),'question':f'Question {i}?','golden_answers':['yes'],'data_source':'nq'} for i in range(12)]
    def test_disjoint_splits_and_heldout_filter(self):
        held=[{'id':'h','question':'Question 0?'}]
        train,val=split_records(self.rows(),held,6,3,42)
        assert_splits(train,val,held)
        self.assertEqual(len(train),6); self.assertEqual(len(val),3)
        self.assertNotIn('Question 0?',[x['question'] for x in train+val])
    def test_training_cannot_include_heldout(self):
        tr=[{'id':'x','question':'Question?','answers':['yes']}]
        with self.assertRaises(ValueError): assert_splits(tr,[],tr)
    def test_split_is_deterministic(self):
        self.assertEqual(split_records(self.rows(),[],6,3,42),split_records(self.rows(),[],6,3,42))
    def test_duplicate_questions_removed(self):
        rows=self.rows()+self.rows()
        tr,val=split_records(rows,[],6,3,42)
        self.assertEqual(len({r['question'] for r in tr+val}),9)
    def test_insufficient_examples_rejected(self):
        with self.assertRaises(ValueError): split_records(self.rows(),[],12,1,42)
    def test_bad_group_size(self):
        with self.assertRaises(ValueError): Config(group_size=1).validate()
    def test_invalid_steps(self):
        with self.assertRaises(ValueError): Config(steps=0).validate()
