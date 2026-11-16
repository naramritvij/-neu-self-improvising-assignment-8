import unittest
from search_r1_team.metrics import normalize_answer, exact_match, extract_answer

class MetricsTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer('  The BLUE, box! '),'blue box')
    def test_alias(self):
        self.assertEqual(exact_match('N.Y.C.',['New York City','NYC']),1)
    def test_not_substring(self):
        self.assertEqual(exact_match('It is Blue City',['Blue City']),0)
    def test_empty(self):
        self.assertEqual(exact_match('',['Blue City']),0)
    def test_bad_labels(self):
        for labels in [[], '', ['']]:
            with self.assertRaises(ValueError): exact_match('x',labels)
    def test_tagged(self):
        self.assertEqual(extract_answer('<think>x</think><answer>Blue City</answer>',tagged=True),'Blue City')
    def test_last_tag(self):
        self.assertEqual(extract_answer('<answer>x</answer><answer>y</answer>',tagged=True),'y')
    def test_missing_tag(self):
        self.assertEqual(extract_answer('Blue City',tagged=True),'')
    def test_direct(self):
        self.assertEqual(extract_answer(' Blue City ',tagged=False),'Blue City')

if __name__ == '__main__': unittest.main()
