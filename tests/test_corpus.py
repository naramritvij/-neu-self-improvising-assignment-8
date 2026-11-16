import json
from pathlib import Path
import tempfile
import unittest
from search_r1_team.corpus import IndexedCorpus

class TestCorpus(unittest.TestCase):
    def test_order_and_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corpus.jsonl'
            p.write_text('\n'.join(json.dumps({'id':str(i),'contents':f'Title\nText {i}'}) for i in range(3)))
            c=IndexedCorpus(p)
            try:
                self.assertEqual(len(c),3); self.assertEqual(c[2]['id'],'2'); self.assertEqual(c[0]['id'],'0')
                with self.assertRaises(IndexError): _=c[-1]
            finally: c.close()
    def test_rebuild_after_source_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.jsonl'; p.write_text('{"id":"0","contents":"x"}\n')
            c=IndexedCorpus(p); c.close()
            with p.open('a') as f: f.write('{"id":"1","contents":"y"}\n')
            c=IndexedCorpus(p)
            try: self.assertEqual(len(c),2)
            finally: c.close()
