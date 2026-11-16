"""SYNTHETIC CPU-only fixtures. Never use these models/results for the assignment."""
from contextlib import nullcontext
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

class CharTokenizer:
    chat_template=None; eos_token_id=0; pad_token_id=0
    def encode(self,text,**kwargs): return [ord(x) for x in text]
    def decode(self,ids,**kwargs): return ''.join(chr(x) for x in ids if x)

class ScriptedPolicy:
    tokenizer=CharTokenizer()
    def __init__(self,outputs): self.outputs=list(outputs)
    def generate(self,ids,**kwargs):
        result=self.outputs.pop(0)
        assert len(result)<=kwargs['max_new_tokens'], 'Test fixture exceeds the declared generation budget'
        return self.tokenizer.encode(result)
    def is_eos(self,x): return x==0

class TinyAdapterPolicy:
    """Frozen GRU backbone and separate rank-2 actor/critic residual adapters."""
    def __init__(self,ppo=False):
        torch.manual_seed(7)
        self.embedding=nn.Embedding(128,8); self.body=nn.GRU(8,8,batch_first=True); self.head=nn.Linear(8,128)
        for mod in (self.embedding,self.body,self.head):
            for p in mod.parameters(): p.requires_grad_(False)
        self.a=nn.Parameter(torch.randn(2,8)*.1); self.b=nn.Parameter(torch.zeros(8,2))
        self.ca=nn.Parameter(torch.randn(2,8)*.1); self.cb=nn.Parameter(torch.zeros(8,2))
        self.v=nn.Linear(8,1); nn.init.zeros_(self.v.weight); nn.init.zeros_(self.v.bias)
        self.ppo=ppo
    def actor_parameters(self): return [self.a,self.b]
    def critic_parameters(self): return [self.ca,self.cb,*self.v.parameters()] if self.ppo else []
    def frozen_parameters(self): return [p for m in (self.embedding,self.body,self.head) for p in m.parameters()]
    def h(self,tr):
        ids=torch.tensor([tr.token_ids]); h,_=self.body(self.embedding(ids))
        pos=torch.tensor(tr.action_positions)
        return h[0,pos-1],ids[0,pos]
    def logprobs(self,tr,reference=False):
        h,targets=self.h(tr)
        if not reference: h=h+h@self.a.T@self.b.T
        return -F.cross_entropy(self.head(h),targets,reduction='none')
    def values(self,tr):
        h,_=self.h(tr); return self.v(h+h@self.ca.T@self.cb.T).squeeze(-1)
