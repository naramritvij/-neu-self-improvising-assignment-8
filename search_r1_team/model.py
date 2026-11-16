"""Hugging Face backend: frozen Qwen base, actor LoRA, and optional critic LoRA.

The reference is the frozen base with adapters disabled. Actor and critic share
base storage but have distinct adapters and optimizers. Log probabilities are
computed only at sampled action positions, in chunks to limit vocabulary logits.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config

class HFPolicy:
    def __init__(self,cfg: Config,*,adapter_path: str|None=None,training: bool=True,device: str='cuda'):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import LoraConfig,get_peft_model,PeftModel
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required for this Qwen-3B run. CPU unit tests do not need one.')
        self.cfg,self.device,self.training=cfg,torch.device(device),training
        dtype=torch.bfloat16 if device=='cuda' else torch.float32
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.model_id,revision=cfg.model_revision)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token=self.tokenizer.eos_token
        base=AutoModelForCausalLM.from_pretrained(cfg.model_id,revision=cfg.model_revision,
                 torch_dtype=dtype,attn_implementation='sdpa',trust_remote_code=False)
        if getattr(base.config,'model_type',None)!='qwen2':
            raise ValueError('This backend is intentionally restricted to the assignment Qwen2/Qwen2.5 family')
        self.resolved_revision=getattr(base.config,'_commit_hash',cfg.model_revision)
        if training:
            if adapter_path: raise ValueError('Use --resume to restore a complete optimizer/RNG checkpoint, not an adapter-only restart')
            lora=LoraConfig(r=cfg.lora_rank,lora_alpha=cfg.lora_alpha,lora_dropout=0.,
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM')
            self.lm=get_peft_model(base,lora,adapter_name='actor')
            if cfg.algorithm=='ppo': self.lm.add_adapter('critic',lora)
        elif adapter_path:
            self.lm=PeftModel.from_pretrained(base,adapter_path,adapter_name='actor',is_trainable=False)
        else:
            self.lm=base
        self.has_adapters = isinstance(self.lm, PeftModel)
        self.lm.to(self.device)
        # PPO needs deterministic old/new log-probability comparisons; no dropout.
        for module in self.lm.modules():
            if isinstance(module,nn.Dropout): module.p=0.
        base.config.use_cache=not training
        if training and cfg.gradient_checkpointing:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.lm.enable_input_require_grads()
        self.value_head=None
        if training and cfg.algorithm=='ppo':
            self.value_head=nn.Linear(base.config.hidden_size,1,device=self.device,dtype=torch.float32)
            nn.init.zeros_(self.value_head.weight); nn.init.zeros_(self.value_head.bias)
        self.eos_ids=set()
        for x in (self.tokenizer.eos_token_id,base.generation_config.eos_token_id):
            if isinstance(x,int): self.eos_ids.add(x)
            elif isinstance(x,list): self.eos_ids.update(x)
        self.lm.eval()

    def actor_parameters(self):
        return [p for n,p in self.lm.named_parameters() if '.actor.' in n and 'lora_' in n]

    def critic_parameters(self):
        result=[p for n,p in self.lm.named_parameters() if '.critic.' in n and 'lora_' in n]
        return result+(list(self.value_head.parameters()) if self.value_head is not None else [])

    def _activate(self,name):
        # Plain Transformers models also expose set_adapter through a mixin.
        # It raises when no adapter is loaded (direct/released inference).
        if self.has_adapters: self.lm.set_adapter(name)
        self.lm.train(self.training and torch.is_grad_enabled())

    def _backbone(self):
        base=self.lm.get_base_model() if self.has_adapters else self.lm
        return base.model,base.lm_head

    def _hidden(self,trajectory):
        ids=torch.tensor([trajectory.token_ids],device=self.device,dtype=torch.long)
        positions=torch.tensor(trajectory.action_positions,device=self.device,dtype=torch.long)
        body,_=self._backbone()
        hidden=body(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
        # Causal LM position i-1 predicts action token i, including after a retrieval.
        return hidden[0,positions-1],ids[0,positions]

    def logprobs(self,trajectory,reference=False):
        self._activate('actor')
        context=self.lm.disable_adapter() if reference and self.has_adapters else nullcontext()
        with context:
            hidden,targets=self._hidden(trajectory)
            _,head=self._backbone()
            outputs=[]
            for h,t in zip(hidden.split(64),targets.split(64)):
                # Recompute the large vocabulary projection in backward instead of storing it.
                def project(x,y): return -F.cross_entropy(head(x).float(),y,reduction='none')
                if torch.is_grad_enabled() and h.requires_grad:
                    from torch.utils.checkpoint import checkpoint
                    outputs.append(checkpoint(project,h,t,use_reentrant=False))
                else: outputs.append(project(h,t))
            return torch.cat(outputs)

    def values(self,trajectory):
        if self.value_head is None: raise ValueError('No PPO critic exists')
        self._activate('critic')
        hidden,_=self._hidden(trajectory)
        return self.value_head(hidden.float()).squeeze(-1)

    def is_eos(self,token): return token in self.eos_ids

    @torch.no_grad()
    def generate(self,ids,*,max_new_tokens,sample,stop_on_search):
        from transformers import StoppingCriteria,StoppingCriteriaList
        self._activate('actor'); self.lm.eval()
        tokenizer=self.tokenizer; start=len(ids)
        class Tags(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                text=tokenizer.decode(input_ids[0,start:],skip_special_tokens=True)
                stop='</answer>' in text or (stop_on_search and '</search>' in text)
                return torch.tensor([stop],device=input_ids.device)
        inp=torch.tensor([ids],device=self.device)
        extra={'temperature':1.,'top_p':1.,'top_k':0} if sample else {}
        result=self.lm.generate(input_ids=inp,attention_mask=torch.ones_like(inp),
              max_new_tokens=max_new_tokens,do_sample=sample,**extra,
              repetition_penalty=1.,no_repeat_ngram_size=0,
              eos_token_id=sorted(self.eos_ids),pad_token_id=self.tokenizer.pad_token_id,
              stopping_criteria=StoppingCriteriaList([Tags()]),use_cache=True)
        return result[0,start:].tolist()

    def save(self,path):
        path=Path(path); path.mkdir(parents=True,exist_ok=True)
        self._activate('actor')
        # Named PEFT adapters are stored under actor/ and critic/ by save_pretrained.
        self.lm.save_pretrained(path,safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        if self.value_head is not None: torch.save(self.value_head.state_dict(),path/'value_head.pt')
        (path/'model_metadata.json').write_text(json.dumps({
            'base_model':self.cfg.model_id,'base_revision':self.resolved_revision,
            'actor_subdirectory':'actor','critic_subdirectory':'critic' if self.value_head else None},indent=2))

    def load_training(self,path):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        path=Path(path)
        names=['actor']+(['critic'] if self.value_head is not None else [])
        for name in names:
            state=load_file(str(path/name/'adapter_model.safetensors'))
            set_peft_model_state_dict(self.lm,state,adapter_name=name)
        if self.value_head is not None:
            self.value_head.load_state_dict(torch.load(path/'value_head.pt',map_location=self.device,weights_only=True))
