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

## Paper-parity changes (branch `ghouse/paper-parity`)

Upstream references are to `PeterGriffinJin/Search-R1` at commit `598e61b` (2025-11-13), the
paper arXiv:2503.09516v5 (Aug 2025) and `RUC-NLPIR/FlashRAG` at `7926ac3`.

| Change | Now | Upstream source |
|---|---|---|
| Search prompt | Verbatim upstream template, question stripped with `?` appended, one user message through the chat template | `scripts/data_process/qa_search_test_merge.py:32-36` (prompt), `:74-76` (`?`); `verl/utils/dataset/rl_dataset.py:128-129` (chat template) |
| Turn logic | Up to 4 action turns of up to 500 tokens; each turn's action is its first `<search>` or `<answer>` tag; an answer ends the episode | `scripts/nq_hotpotqa/v0.2/train_*.sh` (`max_turns=4`, `max_response_length=500`); `search_r1/llm_agent/generation.py:234`, `:420-428` |
| Final turn | After `max_turns`, one more turn with search disabled, so the model can still answer | `generation.py:278-297` |
| Invalid actions | A turn with no valid action receives the upstream message verbatim (masked, not trained on) and the model retries | `generation.py:396-398` |
| Direct inference | FlashRAG no-retrieval prompt: system "Answer the question based on your own knowledge. Only give me the answer and do not output any other words.", user "Question: {q}"; greedy, 32 new tokens | FlashRAG `examples/methods/run_exp.py:38`, `flashrag/prompt/base_prompt.py:12`, `flashrag/config/basic_config.yaml:112-114`. The paper does not print its direct prompt; this is the most likely source. |
| Minibatch updates | Each batch is shuffled into 2 minibatches, one optimizer step each, so the probability ratio moves and clipping is active | Paper App. B.2 (batch 512, mini-batch 256); `ppo_trainer.yaml` `ppo_epochs: 1` |
| GAE | lambda = 1.0, gamma = 1.0 | Paper App. B.2; `verl/trainer/config/ppo_trainer.yaml:153-154` |
| Value clip | 0.5 | `ppo_trainer.yaml:126` (`cliprange_value`) |
| GRPO group | 5 samples per prompt | Paper App. B.2; `v0.2/train_grpo.sh` (`n_agent=5`) |
| Critic warm-up | 0 steps | `v0.2/train_ppo.sh` (`trainer.critic_warmup=0`) |
| Sampling | Temperature 1.0, top-p 1.0 for both algorithms (unchanged) | Paper App. B.2 ("temperature of 1.0 and a top-p value of 1.0"). The v0.2 GRPO script inherits the yaml default 0.95; the paper text is followed. |
| Retrieval retry | Up to 5 retries with exponential backoff on timeouts, connection errors and HTTP 429/5xx; other errors still fail loudly | Robustness addition, not an upstream behaviour change |
| Bamboogle | FlashRAG `bamboogle/test.jsonl` (125 questions) at a pinned revision | `qa_search_test_merge.py:57`. The course sheet has the same 125 questions in the same order, plus one extra alias ("Mildred Gale"). |
| Released checkpoints | `...-qwen2.5-3b-it-em-{grpo,ppo}-v0.2` | `docs/experiment_log.md`: v0.2 is the release linked to arXiv:2503.09516; v0.3 belongs to arXiv:2505.15117 |

Kept as they were, by decision: per-sequence loss averaging (verl averages per token), the
passage formatting and HTML escaping, the 768-token observation cap (upstream 500, wrappers
included), and no entropy bonus (verl default 0.001). An empty `<search></search>` query is treated
as an invalid action instead of being sent to the retriever.

## Remaining declared differences

| Dimension | This implementation |
|---|---|
| Framework | Sequential custom PyTorch/PEFT, not original veRL |
| Fine-tuning | Frozen base, LoRA actor; PPO critic adapter and small value head (upstream: full fine-tuning of separate actor and critic models) |
| Learning rates | LoRA: 2e-5 actor, 1e-4 critic, no warm-up (upstream full fine-tuning: 1e-6 / 1e-5 with warm-up 0.285 / 0.015) |
| Dropout | Zero for deterministic old/new log-probability comparisons |
| Training data | 4,000-example NQ/HotpotQA subset; 128 validation examples (upstream: full merged train set) |
| Budget | Set in the Phase 3 proposal; far below the paper's 512 prompts per step |
| Decoding | Temperature 1 / top-p 1 for training; greedy evaluation (upstream validation uses `do_sample: False`, `ray_trainer.py:478`) |
| Parsing | Last complete answer tag in response-only text (equivalent to upstream `qa_em.py`, which skips the prompt's example tag) |
| Dependencies | Pinned compact stack separate from newer class starter pins |

PPO includes reference KL as per-action reward shaping before GAE. GRPO adds a
nonnegative sampled reference-KL estimator to its clipped actor objective.
GRPO advantages use population standard deviation within each prompt group.
GAE skips observation tokens and uses zero bootstrap at finite-budget termination; with
lambda = gamma = 1 skipping them does not change the advantages. Losses average per trajectory
then batch; no entropy bonus is added. These choices
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
