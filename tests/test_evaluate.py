import unittest
from scripts.evaluate import score_records

class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gold = [{'id':'a','answers':['Blue City']},{'id':'b','answers':['Red City']}]
    def test_complete(self):
        result = score_records(self.gold,[{'id':'a','prediction':'Blue City'},{'id':'b','prediction':'wrong'}],method='direct')
        self.assertEqual(result['em_percent'],50)
        self.assertEqual(result['coverage'],'complete')
    def test_missing(self):
        with self.assertRaises(ValueError): score_records(self.gold,[{'id':'a','prediction':'Blue City'}],method='direct')
    def test_partial(self):
        result = score_records(self.gold,[{'id':'a','prediction':'Blue City'}],method='direct',allow_partial=True)
        self.assertEqual(result['coverage'],'partial_diagnostic')
        self.assertEqual(result['missing_predictions'],1)
    def test_unknown_id(self):
        with self.assertRaises(ValueError): score_records(self.gold,[{'id':'z','prediction':'X'}],method='direct',allow_partial=True)
    def test_tags_required(self):
        result = score_records(self.gold,[{'id':'a','prediction':'Blue City'},{'id':'b','prediction':'<answer>Red City</answer>'}],method='grpo')
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['invalid_or_empty_answers'],1)
    def test_duplicate_predictions(self):
        with self.assertRaises(ValueError): score_records(self.gold,[{'id':'a','prediction':'x'},{'id':'a','prediction':'x'}],method='ppo')
    def test_bad_prediction(self):
        with self.assertRaises(ValueError): score_records(self.gold,[{'id':'a','prediction':None}],method='ppo',allow_partial=True)
    def test_failed_attempt_not_skipped(self):
        result = score_records(self.gold,[{'id':'a','prediction':''},{'id':'b','prediction':'Red City'}],method='direct')
        self.assertEqual(result['evaluated_questions'],2)
        self.assertEqual(result['em'],0.5)

if __name__ == '__main__': unittest.main()
