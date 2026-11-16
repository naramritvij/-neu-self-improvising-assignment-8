import json
from pathlib import Path
import tempfile
import unittest
from scripts.build_report import collect
from search_r1_team.runtime import sha256

class TestReport(unittest.TestCase):
    def fixture(self,root,method='direct'):
        p=Path(root)/method; p.mkdir()
        pred=p/'predictions.jsonl'; pred.write_text('{"id":"synthetic","prediction":"toy"}\n')
        metrics={'method':method.split('_')[-1],'coverage':'complete','predictions_sha256':sha256(pred),
          'expected_questions':1,'evaluated_questions':1,'gold_sha256':'synthetic','em_percent':0,'correct':0}
        manifest={'status':'completed','coverage':'full','total_questions':1,
          'signature':{'method':method,'questions_sha256':'synthetic','retrieval':None,'own_training':{'run_kind':'experiment','step':3}}}
        (p/'metrics.json').write_text(json.dumps(metrics)); (p/'manifest.json').write_text(json.dumps(manifest))
        return p,metrics,manifest
    def test_no_runs_gives_no_measured_scores(self):
        with tempfile.TemporaryDirectory() as d: self.assertEqual(collect(d),[])
    def test_partial_scores_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,r=self.fixture(d); m['coverage']='partial_diagnostic'; (p/'metrics.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): collect(d)
    def test_changed_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,m,r=self.fixture(d); (p/'predictions.jsonl').write_text('changed')
            with self.assertRaises(ValueError): collect(d)
    def test_synthetic_complete_fixture_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d); self.assertEqual(len(collect(d)),1)
    def test_different_gold_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d); p,m,r=self.fixture(d,'own_grpo'); m['gold_sha256']='other'; (p/'metrics.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError): collect(d)
