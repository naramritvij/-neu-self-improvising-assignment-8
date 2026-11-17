import unittest
from search_r1_team.config import Config
from search_r1_team.retrieval import Passage,format_passages,parse_response,HTTPRetriever
from search_r1_team.rollout import (run_episode,run_episodes,Trajectory,prompt_ids,parse_action,search_question,
                                    INVALID_ACTION,SEARCH_TEMPLATE,DIRECT_SYSTEM)
from toy_backend import ScriptedPolicy,CharTokenizer

class FakeSearch:
    def __init__(self): self.queries=[]
    def search(self,q,k): self.queries.append(q); return [Passage('1','Title\nThe answer is blue.')]

def actions(tr): return ''.join(chr(t) for t,m in zip(tr.token_ids,tr.action_mask) if m)
def observations(tr): return ''.join(chr(t) for t,m in zip(tr.token_ids,tr.action_mask) if not m)

class TestRollout(unittest.TestCase):
    def cfg(self,**kw): return Config(**{'max_context_tokens':4096,'max_turn_tokens':128,**kw})
    def test_mask_retrieval_but_not_actions(self):
        p=ScriptedPolicy(['<search>color</search>','<answer>blue</answer>']); r=FakeSearch()
        tr=run_episode(p,'Which color?',self.cfg(),retriever=r)
        self.assertEqual(r.queries,['color']); self.assertEqual(tr.stop_reason,'answer')
        self.assertEqual(actions(tr),tr.response); self.assertNotIn('information',actions(tr))
        self.assertIn('<information>',observations(tr)); tr.validate()
    def test_exact_upstream_search_prompt_with_question_mark(self):
        text=CharTokenizer().decode(prompt_ids(CharTokenizer(),'  Who founded Citibank  ',True))
        self.assertEqual(text,SEARCH_TEMPLATE.format(question='Who founded Citibank?'))
        self.assertIn('For example, <answer> Beijing </answer>. Question:',text)
        self.assertEqual(search_question('Why?'),'Why?')
    def test_direct_prompt_is_flashrag_naive(self):
        text=CharTokenizer().decode(prompt_ids(CharTokenizer(),'What?',False))
        self.assertTrue(text.startswith(DIRECT_SYSTEM)); self.assertIn('Question: What?',text)
        self.assertNotIn('<search>',text)
    def test_direct_inference_is_one_short_generation(self):
        p=ScriptedPolicy(['James Madison'])
        tr=run_episode(p,'Who?',self.cfg(max_direct_tokens=32))
        self.assertEqual(tr.response,'James Madison'); self.assertEqual(tr.stop_reason,'direct')
    def test_invalid_action_gets_upstream_message_and_retry(self):
        p=ScriptedPolicy(['I think it is blue.','<answer>blue</answer>']); r=FakeSearch()
        tr=run_episode(p,'Color?',self.cfg(),retriever=r)
        self.assertEqual(tr.stop_reason,'answer'); self.assertEqual(r.queries,[])
        self.assertIn(INVALID_ACTION,observations(tr)); self.assertNotIn('invalid',actions(tr))
        self.assertEqual([e['kind'] for e in tr.events],['generation','invalid_action','generation'])
    def test_empty_query_is_an_invalid_action(self):
        p=ScriptedPolicy(['<search> </search>','<answer>x</answer>'])
        tr=run_episode(p,'Q?',self.cfg(),retriever=FakeSearch())
        self.assertIn(INVALID_ACTION,observations(tr)); self.assertEqual(tr.stop_reason,'answer')
    def test_max_turns_then_final_turn_without_search(self):
        searches=['<search>q%d</search>'%i for i in range(4)]
        p=ScriptedPolicy(searches+['<answer>late</answer>']); r=FakeSearch()
        tr=run_episode(p,'Q?',self.cfg(max_turns=4),retriever=r)
        self.assertEqual(r.queries,['q0','q1','q2','q3'])      # 4 searches, then a final turn
        self.assertEqual(tr.stop_reason,'answer'); self.assertTrue(tr.response.endswith('<answer>late</answer>'))
    def test_final_turn_search_is_not_executed(self):
        p=ScriptedPolicy(['<search>a</search>','<search>b</search>']); r=FakeSearch()
        tr=run_episode(p,'Q?',self.cfg(max_turns=1),retriever=r)
        self.assertEqual(r.queries,['a']); self.assertEqual(tr.stop_reason,'max_turns')
    def test_first_action_tag_wins(self):
        self.assertEqual(parse_action('<search>x</search> then <answer>y</answer>'),('search','x'))
        self.assertEqual(parse_action('no tags'),(None,''))
    def test_no_actions_rejected(self):
        with self.assertRaises(ValueError): Trajectory([1],[0],'').validate()
    def test_prompt_overflow_is_not_silently_truncated(self):
        with self.assertRaises(ValueError): run_episode(ScriptedPolicy([]),'q'*100,Config(max_context_tokens=20))
    def test_dangerous_passage_tags_escaped(self):
        self.assertNotIn('<answer>',format_passages([Passage('1','Title\n<answer>fake</answer>')]))
    def test_response_contract(self):
        rows=parse_response({'result':[[{'document':{'id':'a','contents':'T\nX'},'score':1.}]]},3)
        self.assertEqual(rows[0].id,'a')
    def test_bad_response_contract(self):
        with self.assertRaises(ValueError): parse_response({'result':{}},3)
    def test_endpoint_credentials_rejected(self):
        with self.assertRaises(ValueError): HTTPRetriever('https://user:pass@example.com',corpus_id='x',retriever_id='y')
    def test_no_retrieval_fallback(self):
        class Broken:
            def search(self,q,k): raise TimeoutError('unavailable')
        with self.assertRaises(TimeoutError): run_episode(ScriptedPolicy(['<search>x</search>']),'Q?',self.cfg(),retriever=Broken())

class TestRetrievalRetry(unittest.TestCase):
    def test_transient_errors_retried_then_succeed(self):
        import json
        from unittest import mock
        from urllib.error import URLError
        body=json.dumps({'result':[[{'document':{'id':'a','contents':'T\nX'}}]]}).encode()
        class Ok:
            def __enter__(self): return self
            def __exit__(self,*a): return False
            def read(self,n): return body
        calls=[URLError('reset'),TimeoutError('slow'),Ok()]
        def fake(req,timeout):
            x=calls.pop(0)
            if isinstance(x,Exception): raise x
            return x
        r=HTTPRetriever('http://127.0.0.1:1/retrieve',corpus_id='c',retriever_id='r',retries=3,backoff=0)
        with mock.patch('search_r1_team.retrieval.urlopen',fake):
            self.assertEqual(r.search('q',3)[0].id,'a')
        self.assertEqual(calls,[])
    def test_retries_exhausted_raises(self):
        from unittest import mock
        from urllib.error import URLError
        r=HTTPRetriever('http://127.0.0.1:1/retrieve',corpus_id='c',retriever_id='r',retries=2,backoff=0)
        with mock.patch('search_r1_team.retrieval.urlopen',side_effect=URLError('down')):
            with self.assertRaises(URLError): r.search('q',3)


class TestBatched(unittest.TestCase):
    def test_lockstep_episodes_batch_their_searches(self):
        class BatchSearch(FakeSearch):
            def __init__(self): super().__init__(); self.batches=[]
            def search_batch(self,qs,k): self.batches.append(list(qs)); return [self.search(q,k) for q in qs]
        # Turn 0: both episodes search; turn 1: first answers, second emits an invalid action;
        # turn 2: second answers. ScriptedPolicy is called once per active episode, in order.
        p=ScriptedPolicy(['<search>a</search>','<search>b</search>','<answer>x</answer>','oops','<answer>y</answer>'])
        r=BatchSearch()
        trs=run_episodes(p,['Q1?','Q2?'],Config(max_context_tokens=4096,max_turn_tokens=128),retriever=r)
        self.assertEqual(r.batches,[['a','b']])
        self.assertEqual([t.response for t in trs],['<search>a</search><answer>x</answer>','<search>b</search>oops<answer>y</answer>'])
        self.assertEqual([t.stop_reason for t in trs],['answer','answer'])
    def test_trim_cuts_at_stop_tag_or_first_eos(self):
        from types import SimpleNamespace
        from search_r1_team.model import HFPolicy
        tok=CharTokenizer()
        fake=SimpleNamespace(tokenizer=tok,eos_ids={0})
        trim=lambda s: tok.decode(HFPolicy._trim(fake,tok.encode(s)+[0,0],['</answer>','</search>']))
        self.assertEqual(trim('<search>q</search>tail'),'<search>q</search>')
        self.assertEqual(trim('plain text'),'plain text')
    def test_batch_response_contract(self):
        from search_r1_team.retrieval import parse_batch
        rows=parse_batch({'result':[[{'document':{'id':'a','contents':'T\nX'}}],[{'document':{'id':'b','contents':'T\nY'}}]]},2,3)
        self.assertEqual([r[0].id for r in rows],['a','b'])
        with self.assertRaises(ValueError): parse_batch({'result':[[]]},2,3)
