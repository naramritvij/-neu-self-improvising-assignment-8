"""Token-exact search trajectories; only model-sampled tokens receive policy loss."""
from dataclasses import dataclass,field
import re
from .config import Config
from .retrieval import format_passages

SEARCH_PROMPT = (
 'Answer the given question. Think inside <think> and </think>. '
 'When external knowledge is needed, emit <search>your query</search>. '
 'The environment returns passages in <information>...</information>. '
 'Use these passages to continue reasoning; do not write information tags yourself. '
 'Finish with a concise answer inside <answer>...</answer>. Question: '
)
DIRECT_PROMPT = 'Answer the question with only the short answer, without explanation. Question: '

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


def prompt_ids(tokenizer,question: str,search: bool) -> list[int]:
    if not isinstance(question,str) or not question.strip(): raise ValueError('Empty question')
    message=(SEARCH_PROMPT if search else DIRECT_PROMPT)+question.strip()
    if getattr(tokenizer,'chat_template',None):
        return tokenizer.apply_chat_template([{'role':'user','content':message}],tokenize=True,add_generation_prompt=True)
    return tokenizer.encode(message,add_special_tokens=True)


def run_episode(policy,question: str,cfg: Config,*,retriever=None,sample: bool=True) -> Trajectory:
    tokenizer=policy.tokenizer
    ids=list(prompt_ids(tokenizer,question,retriever is not None))
    if len(ids)>=cfg.max_context_tokens-1: raise ValueError('Prompt exceeds the context budget')
    mask=[0]*len(ids); outputs=[]; events=[]; used=0; searches=0; reason='token_budget'; pending=''
    # Observation and prompt tokens stay in the causal context but not the action list.
    while used<cfg.max_action_tokens:
        room=cfg.max_context_tokens-len(ids)
        if room<=0: reason='context_budget'; break
        n=min(cfg.max_turn_tokens,cfg.max_action_tokens-used,room)
        # Stop tags are interpreted only in the newly generated segment, never in the prompt.
        new=policy.generate(ids,max_new_tokens=n,sample=sample,stop_on_search=retriever is not None)
        if not new: reason='empty_generation'; break
        text=tokenizer.decode(new,skip_special_tokens=True)
        ids.extend(new); mask.extend([1]*len(new)); used+=len(new); outputs.append(text); pending+=text
        events.append({'kind':'generation','text':text,'tokens':len(new)})
        if re.search(r'<answer>[\s\S]*?</answer>',pending): reason='answer'; break
        # A tag may start in the preceding token-budget segment. Never reuse an old search.
        query=re.findall(r'<search>(.*?)</search>',pending,re.DOTALL)
        if retriever is not None and query:
            if searches>=cfg.max_searches: reason='search_budget'; break
            q=query[-1].strip()
            if not q: reason='empty_query'; break
            passages=retriever.search(q,cfg.top_k) # Fail loudly; never fabricate a passage.
            searches+=1
            pending=''
            raw=format_passages(passages)
            prefix=tokenizer.encode('\n<information>',add_special_tokens=False)
            suffix=tokenizer.encode('</information>\n',add_special_tokens=False)
            # Reserve at least one policy action after the observation.
            room=cfg.max_context_tokens-len(ids)-len(prefix)-len(suffix)-1
            if room<=0: reason='context_budget'; break
            content=tokenizer.encode(raw,add_special_tokens=False)[:min(room,cfg.max_observation_tokens)]
            obs=prefix+content+suffix
            ids.extend(obs); mask.extend([0]*len(obs))
            events.append({'kind':'retrieval','query':q,'passage_ids':[p.id for p in passages],
                           'contents':[p.contents for p in passages],'inserted_tokens':len(obs),
                           'inserted_text':tokenizer.decode(obs,skip_special_tokens=True)})
        elif policy.is_eos(new[-1]): reason='eos'; break
        elif len(new)<n: reason='stopped'; break
    return Trajectory(ids,mask,''.join(outputs),events,reason).validate()
