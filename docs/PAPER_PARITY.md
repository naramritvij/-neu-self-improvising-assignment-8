# Paper parity, lineage and deviations

## Class starting point

The inspected starter is `suhabe/tiny-zero-lora` at commit
`3659acfad61dab2f90fa46294566ff6808452e87`. Its train.py uses Qwen2.5-3B-Instruct,
Countdown arithmetic, rank-16/alpha-32 LoRA, seven Qwen projection targets, GRPO
and format/equation rewards. The fetcher preserves that source. The runner uses
a separate environment and explicit output/budget overrides. A two-step smoke
run is not the full starter reproduction.

The QA adaptation keeps the model family and LoRA configuration but changes the
arithmetic task into search-based QA with answer-correctness rewards. Explicit
PyTorch loops replace TRL orchestration to expose observation masks and PPO
actor/critic handling. Do not claim the starter already implements PPO or search.

## Declared differences

| Dimension | This implementation |
|---|---|
| Framework | Sequential custom PyTorch/PEFT, not original veRL |
| Fine-tuning | Frozen base, LoRA actor; PPO critic adapter and small value head |
| Dropout | Zero for deterministic old/new log-probability comparisons |
| Training data | Proposed 4,000-example NQ/HotpotQA subset; 128 validation examples |
| Budget | Proposed 500 steps, 4 prompts × 8 rollouts; separate two-step smoke profiles |
| PPO critic | Shares frozen base storage; distinct adapter and value head |
| Decoding | Temperature 1/unrestricted softmax for training; greedy evaluation |
| Prompts | Compact search and short-answer prompts; not exact FlashRAG baseline parity |
| Limits | Bounded context, observations, action tokens and search count |
| Parsing | Last complete answer tag in response-only text |
| Dependencies | Pinned compact stack separate from newer class starter pins |

PPO includes reference KL as per-action reward shaping before GAE. GRPO adds a
nonnegative sampled reference-KL estimator to its clipped actor objective.
GRPO advantages use population standard deviation within each prompt group.
GAE skips observation tokens and uses zero bootstrap at finite-budget termination.
Losses average per trajectory then batch; no entropy bonus is added. These choices
are not silently presented as exact upstream equivalence.

The retrieved-text context is excluded from action losses. The intended retriever
is top-3 E5 over Wikipedia 2018. The API/format is implemented, but full-index
integration and retrieval parity have not been demonstrated.

## Separate inference-only track

`released_grpo` and `released_ppo` name the authors' 3B **instruct** v0.2 models.
`own_grpo` and `own_ppo` use the team's saved actor adapters over the locked Qwen
instruct base. Evaluating a released full checkpoint is not your own LoRA training.

## Evaluation limitations

Bamboogle is held out from training and model selection. Its questions are used
only to exclude overlap. Upstream qa_em.py expects prompt+response, including an
example answer tag; our scorer consumes response-only text. Normalization is
inspected but end-to-end evaluator parity is not claimed. Single-seed results
and reduced training budgets need explicit qualification.

## Sources

- https://github.com/suhabe/tiny-zero-lora/blob/3659acfad61dab2f90fa46294566ff6808452e87/train.py
- https://arxiv.org/html/2503.09516v5
- https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py
- https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py
- https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py
