import tempfile
import unittest
from pathlib import Path
from search_r1_team.data import load_bamboogle_csv, read_jsonl, write_jsonl

class DataTests(unittest.TestCase):
    def test_columns_alias_and_metadata(self):
        rows = load_bamboogle_csv('Question,,,,Answer,Answer 2,metadata\nWhich fictional city?,,,,Blue City,Azure City,not a prompt\n')
        self.assertEqual(rows[0]['id'],'bamboogle_0001')
        self.assertEqual(rows[0]['answers'],['Blue City','Azure City'])
        self.assertEqual(set(rows[0]),{'id','question','answers'})
    def test_blank_rows(self):
        rows = load_bamboogle_csv('Question,Answer\n,\nQ?,X\n')
        self.assertEqual(rows[0]['id'],'bamboogle_0002')
    def test_bad_header(self):
        with self.assertRaises(ValueError): load_bamboogle_csv('<html>login</html>')
    def test_missing_answer(self):
        with self.assertRaises(ValueError): load_bamboogle_csv('Question,Answer\nQ?,\n')
    def test_bom(self):
        self.assertEqual(len(load_bamboogle_csv('\ufeffQuestion,Answer\nQ?,X\n')),1)
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.jsonl'; rows = [{'id':'a','answers':['X']}]
            write_jsonl(path,rows)
            self.assertEqual(read_jsonl(path),rows)
    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.jsonl'
            write_jsonl(path,[{'id':'a'},{'id':'a'}])
            with self.assertRaises(ValueError): read_jsonl(path)

if __name__ == '__main__': unittest.main()
