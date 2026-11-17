"""Token-exact search trajectories; only model-sampled tokens receive policy loss.

Turn logic follows upstream Search-R1 (PeterGriffinJin/Search-R1 @ 598e61b,
search_r1/llm_agent/generation.py): up to `max_turns` action turns, each ending at the first
</search> or </answer>; an answer ends the episode, a search inserts retrieved passages, and
anything else receives the upstream "invalid action" message. After `max_turns`, one final
turn is generated with search disabled, so the model can still answer.
"""
from dataclasses import dataclass, field
import re
from .config import Config
from .retrieval import format_passages

# Upstream training/evaluation prompt, verbatim: scripts/data_process/qa_search_train_merge.py:32-36
# and qa_search_test_merge.py:32-36 (template_type='base', used for base AND instruct models).
SEARCH_TEMPLATE = (
    'Answer the given question. '
    'You must conduct reasoning inside <think> and </think> first every time you get new information. '
    'After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> '
    'and it will return the top searched results between <information> and </information>. '
    'You can search as many times as your want. '
    'If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, '
    'without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n'
)
# Direct inference: FlashRAG's no-retrieval ("naive") generation prompt
# (RUC-NLPIR/FlashRAG @ 7926ac3, examples/methods/run_exp.py:38; user prompt from
# flashrag/prompt/base_prompt.py:12). The Search-R1 paper does not print its direct prompt.
DIRECT_SYSTEM = 'Answer the question based on your own knowledge. Only give me the answer and do not output any other words.'
DIRECT_USER = 'Question: {question}'
# Upstream invalid-action observation, verbatim: search_r1/llm_agent/generation.py:396-398.
INVALID_ACTION = ('\nMy previous action is invalid. If I want to search, I should put the query between '
                  '<search> and </search>. If I want to give the final answer, I should put the answer '
                  'between <answer> and </answer>. Let me try again.\n')
# Upstream action parser: first <search>...</search> or <answer>...</answer> (generation.py:420-428).
ACTION = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)


@dataclass
class Trajectory:
    token_ids: list[int]
    action_mask: list[int]
    response: str
    events: list[dict] = field(default_factory=list)
    stop_reason: str = 'unknown'

    @property
    def action_positions(self): return [i for i,m in enumerate(self.action_mask) if m]

    def validate(self):
        if len(self.token_ids)!=len(self.action_mask): raise ValueError('Mask length mismatch')
        if not self.token_ids or self.action_mask[0]: raise ValueError('A prompt token must precede the first action')
        if not self.action_positions: raise ValueError('Trajectory has no sampled actions')
        if any(x not in (0,1) for x in self.action_mask): raise ValueError('Invalid action mask')
        return self


def search_question(question: str) -> str:
    # Upstream strips the question and appends '?' when missing (qa_search_test_merge.py:74-76).
    q=question.strip()
    return q if q.endswith('?') else q+'?'


def prompt_ids(tokenizer,question: str,search: bool) -> list[int]:
    if not isinstance(question,str) or not question.strip(): raise ValueError('Empty question')
    if search:
        # Upstream wraps the prefix as a single user message and applies the chat template with
        # add_generation_prompt=True (verl/utils/dataset/rl_dataset.py:128-129).
        messages=[{'role':'user','content':SEARCH_TEMPLATE.format(question=search_question(question))}]
    else:
        messages=[{'role':'system','content':DIRECT_SYSTEM},
                  {'role':'user','content':DIRECT_USER.format(question=question.strip())}]
    if getattr(tokenizer,'chat_template',None):
        return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    return tokenizer.encode('\n'.join(m['content'] for m in messages),add_special_tokens=True)


def parse_action(text: str) -> tuple[str|None,str]:
    match=ACTION.search(text)
    return (match.group(1),match.group(2).strip()) if match else (None,'')


def _generate(policy,contexts,**kw):
    # Batched generation when the policy supports it (HFPolicy); otherwise one context at a time.
    if hasattr(policy,'generate_batch'): return policy.generate_batch(contexts,**kw)
    return [policy.generate(ids,**kw) for ids in contexts]


def _search(retriever,queries,top_k):
    if hasattr(retriever,'search_batch'): return retriever.search_batch(queries,top_k)
    return [retriever.search(q,top_k) for q in queries]


class _Episode:
    def __init__(self,ids):
        self.ids=list(ids); self.mask=[0]*len(self.ids); self.outputs=[]; self.events=[]
        self.reason=None


def run_episodes(policy,questions,cfg: Config,*,retriever=None,sample: bool=True) -> list[Trajectory]:
    """Run many episodes in lockstep: one batched generation per turn for all active episodes and
    one batched retrieval call for all of their queries. Per-episode logic is identical to the
    single-episode version (see module docstring)."""
    tokenizer=policy.tokenizer
    episodes=[]
    for q in questions:
        ids=prompt_ids(tokenizer,q,retriever is not None)
        if len(ids)>=cfg.max_context_tokens-1: raise ValueError('Prompt exceeds the context budget')
        episodes.append(_Episode(ids))
    if retriever is None:
        # Direct inference: one short greedy answer (FlashRAG generation max_tokens=32,
        # flashrag/config/basic_config.yaml:112-114).
        room=min(cfg.max_context_tokens-len(e.ids) for e in episodes)
        news=_generate(policy,[e.ids for e in episodes],max_new_tokens=min(cfg.max_direct_tokens,room),
                       sample=sample,stop_on_search=False)
        out=[]
        for e,new in zip(episodes,news):
            if not new: raise ValueError('Empty generation')
            text=tokenizer.decode(new,skip_special_tokens=True)
            e.ids.extend(new); e.mask.extend([1]*len(new)); e.events.append({'kind':'generation','text':text,'tokens':len(new)})
            out.append(Trajectory(e.ids,e.mask,text,e.events,'direct').validate())
        return out

    for turn in range(cfg.max_turns+1):
        final=turn==cfg.max_turns           # final turn: search disabled (generation.py:278-297)
        active=[e for e in episodes if e.reason is None]
        for e in active:
            if cfg.max_context_tokens-len(e.ids)<=1: e.reason='context_budget'
        active=[e for e in active if e.reason is None]
        if not active: break
        room=min(cfg.max_context_tokens-len(e.ids) for e in active)
        news=_generate(policy,[e.ids for e in active],max_new_tokens=min(cfg.max_turn_tokens,room),
                       sample=sample,stop_on_search=True)
        pending=[]
        for e,new in zip(active,news):
            if not new: e.reason='empty_generation'; continue
            text=tokenizer.decode(new,skip_special_tokens=True)
            e.ids.extend(new); e.mask.extend([1]*len(new)); e.outputs.append(text)
            action,content=parse_action(text)
            e.events.append({'kind':'generation','turn':turn,'text':text,'tokens':len(new),'action':action})
            if action=='answer': e.reason='answer'
            elif final: e.reason='max_turns'   # a final-turn search gets no observation, as upstream
            else: pending.append((e,action,content))
        searches=[(e,content) for e,action,content in pending if action=='search' and content]
        results=_search(retriever,[c for _,c in searches],cfg.top_k) if searches else []
        found={id(e):(content,passages) for (e,content),passages in zip(searches,results)}
        for e,action,content in pending:
            if id(e) in found:
                query,passages=found[id(e)]
                prefix=tokenizer.encode('\n<information>',add_special_tokens=False)
                suffix=tokenizer.encode('</information>\n',add_special_tokens=False)
                body=tokenizer.encode(format_passages(passages),add_special_tokens=False)[:cfg.max_observation_tokens]
                obs=prefix+body+suffix
                event={'kind':'retrieval','query':query,'passage_ids':[x.id for x in passages],
                       'contents':[x.contents for x in passages]}
            else:
                # No valid action (or an empty query): the upstream invalid-action message.
                obs=tokenizer.encode(INVALID_ACTION,add_special_tokens=False)
                event={'kind':'invalid_action'}
            obs=obs[:max(0,cfg.max_context_tokens-len(e.ids)-1)]
            if not obs: e.reason='context_budget'; continue
            e.ids.extend(obs); e.mask.extend([0]*len(obs))
            e.events.append({**event,'turn':turn,'inserted_tokens':len(obs),
                             'inserted_text':tokenizer.decode(obs,skip_special_tokens=True)})
    return [Trajectory(e.ids,e.mask,''.join(e.outputs),e.events,e.reason or 'max_turns').validate()
            for e in episodes]


def run_episode(policy,question: str,cfg: Config,*,retriever=None,sample: bool=True) -> Trajectory:
    return run_episodes(policy,[question],cfg,retriever=retriever,sample=sample)[0]
