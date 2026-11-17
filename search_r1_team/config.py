"""Explicit, serializable settings for the compact single-GPU implementation."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass
class Config:
    algorithm: str = 'grpo'
    model_id: str = 'Qwen/Qwen2.5-3B-Instruct'
    model_revision: str = 'main'
    seed: int = 42
    steps: int = 2
    prompts_per_step: int = 1
    group_size: int = 2
    update_epochs: int = 1
    minibatches: int = 2          # optimizer steps per batch; >=2 so the PPO ratio and clip are active
    learning_rate: float = 2e-5
    critic_learning_rate: float = 1e-4
    clip: float = .2
    value_clip: float = .5        # verl critic.cliprange_value (ppo_trainer.yaml:126)
    kl_beta: float = .001
    gamma: float = 1.
    gae_lambda: float = 1.        # paper App. B.2: GAE lambda=1, gamma=1
    max_grad_norm: float = 1.
    lora_rank: int = 16
    lora_alpha: int = 32
    max_context_tokens: int = 2048
    max_turn_tokens: int = 500    # upstream data.max_response_length=500 per turn
    max_turns: int = 4            # upstream v0.2 max_turns=4, then one final no-search turn
    max_direct_tokens: int = 32   # FlashRAG generation max_tokens=32 for direct inference
    max_observation_tokens: int = 512
    top_k: int = 3
    eval_every: int = 0
    validation_limit: int = 32
    save_every: int = 1
    gradient_checkpointing: bool = True
    critic_warmup_steps: int = 1

    def validate(self):
        if self.algorithm not in {'grpo','ppo'}:
            raise ValueError('algorithm must be grpo or ppo')
        for key in ('steps','prompts_per_step','group_size','update_epochs','minibatches','lora_rank',
                    'lora_alpha','max_context_tokens',
                    'max_turn_tokens','max_direct_tokens','max_observation_tokens','top_k','validation_limit','save_every'):
            if not isinstance(getattr(self,key),int) or getattr(self,key)<1:
                raise ValueError(f'{key} must be a positive integer')
        if self.algorithm=='grpo' and self.group_size<2:
            raise ValueError('GRPO needs at least two completions per prompt')
        if self.max_turns<0 or self.eval_every<0 or self.critic_warmup_steps<0:
            raise ValueError('search/evaluation/warmup counts cannot be negative')
        if not 0 < self.clip < 1 or not 0 < self.value_clip < 1:
            raise ValueError('clipping ranges must lie between zero and one')
        if not 0 <= self.kl_beta or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid KL, discount or GAE parameter')
        if min(self.learning_rate,self.critic_learning_rate,self.max_grad_norm)<=0:
            raise ValueError('learning rates and gradient norm must be positive')
        return self

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))).validate()

    def to_dict(self): return asdict(self)
