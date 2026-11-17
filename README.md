# Search-R1 with LoRA on Bamboogle — INFO 7375 Assignment 8

**Course:** INFO 7375, Self-Improving AI · Week 08 · **Author:** Ghouse (completed individually;
Ritvij initialized this repository, and his code is the base this work builds on)

The assignment: understand and replicate the class starter
[`suhabe/tiny-zero-lora`](https://github.com/suhabe/tiny-zero-lora), then replicate the
Qwen2.5-3B-Instruct rows of Search-R1 ([Jin et al., arXiv:2503.09516v5](https://arxiv.org/abs/2503.09516))
on Bamboogle: Direct Inference, and Search-R1-instruct trained with both PPO and GRPO, using
LoRA (PEFT) instead of full fine-tuning.

> **Status: interim.** The Direct Inference and released-checkpoint rows are final. The LoRA
> GRPO and PPO runs were still training when this was pushed; their rows will be filled in.

## Results (Bamboogle, 125 questions, exact match)

| Method | Model | Ours | Paper | Paper source |
|---|---|---:|---:|---|
| Direct Inference | Qwen2.5-3B-Instruct, no retrieval | **0.024** (3/125) | 0.024 | Table 2 |
| Search-R1-instruct, GRPO (released v0.2 checkpoint) | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.2` | **0.272** (34/125) | 0.232 | Table 3 |
| Search-R1-instruct, PPO (released v0.2 checkpoint) | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo-v0.2` | **0.336** (42/125) | 0.264 | Table 2 |
| Search-R1-instruct, GRPO (our LoRA run) | Qwen2.5-3B-Instruct + rank-16 LoRA | in progress | 0.232 | Table 3 |
| Search-R1-instruct, PPO (our LoRA run) | Qwen2.5-3B-Instruct + rank-16 LoRA actor and critic | in progress | 0.264 | Table 2 |

All rows use the same 125 FlashRAG Bamboogle questions, normalized exact match (lowercase, strip
punctuation and articles, collapse whitespace, any accepted alias), greedy decoding, and for the
search rows the same E5 retriever over Wikipedia-2018 with top-3 passages, up to 4 searches and
then one final turn without search.

- **Direct Inference matches the paper exactly (0.024).** It uses FlashRAG's no-retrieval
  prompt with greedy decoding and a 32-token answer.
- **The released checkpoints score above the paper** (GRPO +0.040, PPO +0.072); the paper's
  ordering, PPO above GRPO, holds. One question is worth 0.008 here, so these gaps are 5 and 9
  questions. Possible reasons: the paper evaluates through veRL's batched vLLM pipeline, while we
  use Hugging Face generation in bf16 with our own turn loop (small numerical and tokenization
  differences change greedy paths); our scorer reads only the model's response; and the
  released v0.2 files may not be the exact step the paper reported. These numbers are therefore
  a faithful re-evaluation of the released models, not a bit-exact reproduction of the table.

## What was run

| Stage | Where | Result |
|---|---|---|
| Starter walkthrough (`walkthrough.py`, tokenizer + rewards) | Local CPU | Prompts and all 7 reward scenarios behave as documented |
| Starter's own `train.py`, 20-step run | Modal A100-80GB | Step-0 held-out eval ran (solve rate 5.5%, format 32.0%); the first training step ran out of GPU memory. See [the starter section](#tiny-zero-lora-replication). |
| E5 index, corpus and encoder download (64.6 GB + 14.4 GB) | Modal CPU | Done; corpus repaired (see below) |
| Direct Inference, 125 questions | Modal L4 | EM 0.024 |
| Released v0.2 GRPO and PPO, 125 questions with search | Modal L40S + co-located E5 | EM 0.272 and 0.336 |
| Our LoRA GRPO and PPO training | Modal L40S + co-located E5 | In progress |

## Method

**Paper parity.** The search prompt, turn logic and invalid-action handling follow upstream
Search-R1 v0.2 line by line ([docs/PAPER_PARITY.md](docs/PAPER_PARITY.md) cites the file and line
for each):

- **Prompt:** the verbatim training template ("Answer the given question. You must conduct
  reasoning inside <think> and </think> first every time you get new information. ... For example,
  <answer> Beijing </answer>. Question: …?").
- **Turns:** up to 4 action turns of up to 500 tokens each. Each turn's first `<search>` or
  `<answer>` tag decides the action; then one final turn with search disabled.
- **Invalid actions:** a turn with no valid action gets upstream's "My previous action is
  invalid…" message and a retry.
- **Loss masking:** retrieved passages and environment messages stay in context but are
  excluded from the policy loss.
- **Reward:** exact match of the last `<answer>` tag only (no format reward), as upstream.

**Retrieval.** E5 (`intfloat/e5-base-v2`) dense retrieval over the 21,015,324-passage
Wikipedia-2018 corpus with the upstream flat FAISS index (`PeterJinGo/wiki-18-e5-index`), top-3.
The retriever runs inside the GPU container (`--retrieval-url local`), so it needs about 96 GiB of
RAM for the index but no separate server. `wiki-18.jsonl.gz` turned out to be a gzipped tar
archive, whose header and padding would have corrupted the first and last passages; the corpus
was repaired and re-verified (21,015,324 rows).

**Training (our LoRA runs).**
- **Model:** frozen Qwen2.5-3B-Instruct with a rank-16, alpha-32 LoRA adapter on all seven
  projections, the starter's LoRA configuration. PPO adds a separate critic LoRA adapter and a
  value head; the reference policy is the base model with adapters disabled.
- **GRPO:** 8 prompts × 5 samples per step, group-normalized advantages, clipped ratio, and a
  `low_var_kl` loss with β = 0.001.
- **PPO:** 40 prompts per step, GAE with λ = γ = 1, KL reward shaping with β = 0.001, and value
  clip 0.5.
- **Both:** two minibatch optimizer steps per batch (so clipping is active), AdamW at lr 2e-5
  (actor) and 1e-4 (critic), sampling at temperature 1 and top-p 1.
- **Generation:** batched across all rollouts of a step.

## Search traces (released PPO checkpoint)

**Correct, two searches** (`bamboogle_0001`, gold: James Madison):

```text
<think>To answer this question, I need to first determine the year Citibank was founded. ...
<search>When was Citibank founded?</search>
  [information: Doc 1 (Citigroup) … Doc 2/3 …]
<think>From the information provided, Citibank was founded in 1812. Now, we need to find out
who was the president of the United States in 1812.
<search>Who was president of the United States in 1812?</search>
  [information: Doc 1 (1812 United States elections) …]
<think>From the information provided, in 1812, James Madison was the President ...</think>
<answer>James Madison</answer>
```

**Wrong, two searches** (`bamboogle_0006`, "When did the president who set the precedent of a
two term limit enter office?", gold: April 30, 1789):

```text
<search>president two-term limit precedent</search>
  [information: Doc 1 (Twenty-second Amendment to the United States Constitution) …]
<think>... it seems Franklin D. Roosevelt became the first president to set the precedent of a
two-term limit. Now, I need to find out when he entered office.
<search>Franklin D. Roosevelt inauguration date</search>
  [information: Doc 1 (First inauguration of Franklin D. Roosevelt) …]
<answer>March 4, 1933</answer>
```

Full traces are in [results/traces/](results/traces/).

## Failure analysis (released checkpoints)

| | Released PPO | Released GRPO |
|---|---:|---:|
| Correct | 42 | 34 |
| Wrong answer after searching | 63 | 80 |
| Near miss (gold and answer contain each other) | 12 | 9 |
| No answer within 4 searches + final turn | 8 | 2 |
| Answered without any search | 0 | 0 |
| Mean searches per question | 2.19 | 2.23 |

- **Wrong bridge entity (most common).** Bamboogle questions are two-hop. When the first search
  returns a related but different entity, the model commits to it and answers the second hop
  correctly for the wrong entity. The two-term-limit question above is typical: the Twenty-second
  Amendment passage leads it to Franklin D. Roosevelt instead of George Washington.
- **Near misses under strict exact match.** Twelve PPO answers differ only in granularity,
  for example "Lockheed" for "Lockheed Corporation", "May 27, 1827" for "1827", and "1891" for
  "January 8, 1891". These count as wrong under EM, as in the paper.
- **Search loops.** PPO more often keeps searching until the turn budget runs out (8
  questions), while GRPO answers earlier (2), which partly offsets its lower accuracy.
- **Retrieval misses.** For several questions the needed fact is not in the top-3 passages, so
  the model guesses after searching (e.g. the Voyager launch-vehicle question).

## tiny-zero-lora replication

**What the starter does.**
- **Task:** Countdown arithmetic, from `Jiayi-Pan/Countdown-Tasks-3to4` (2,000 examples, a 90/10
  split). Reach the target number using each given number exactly once.
- **Model and prompt:** Qwen2.5-3B-Instruct in bf16, with a chat prompt whose assistant turn is
  pre-filled with "Let me solve this step by step.\n<think>".
- **LoRA:** rank 16, alpha 32, dropout 0.05, on q/k/v/o/gate/up/down.
- **GRPO:** TRL's `GRPOTrainer` with 16 generations per prompt, lr 2e-5 (linear decay), β =
  0.001, temperature 1.0, up to 1,024 completion tokens and vLLM generating on the same GPU.
- **Rewards:** a strict format reward plus an equation reward (each number used once,
  evaluated safely, within 1e-5 of the target).

**How this project carries it over.**
- **Kept:** the same base model, the same LoRA rank, alpha and target modules, KL β = 0.001,
  lr 2e-5, temperature 1.0, and the group-normalized advantages with a clipped objective.
- **Changed:** LoRA dropout is 0 instead of 0.05, so old and new log-probabilities can be
  compared exactly; TRL's trainer is replaced by an explicit PyTorch loop, which supports
  multi-turn retrieval with masked observations and a PPO critic; and the reward is answer exact
  match, as in Search-R1.

**Replication runs.**
- **Local walkthrough** (pinned commit `3659acf`, its own pinned environment): both prompts
  render through the Qwen chat template, and all seven reward scenarios score as their labels
  say (correct format and equation 1/1; wrong result 1/0; missing think block 0/1; unused
  number 1/0; duplicate inputs 1/1; division by zero 1/0; missing tags 0/0).
- **Modal 20-step run of its own `train.py`** (A100-80GB, unmodified apart from the output path
  and step count):
  - *First attempt:* crash-looped before training because of a bug in our launcher (an
    import-time path check that only holds locally); fixed with `modal.is_local()`.
  - *Second attempt:* the step-0 held-out evaluation ran (solve rate **5.5%**, format **32.0%**,
    56.6 s). Then the first training step ran **out of GPU memory** (79.2 GiB in use). The starter
    is sized for the B300 (288 GB) in its own `modal_train.py`: 16 generations of 1,024 tokens,
    a training batch of 16, no gradient checkpointing, and vLLM reserving 30% of the GPU.
  - *Not retried tonight.* A B300 costs $7.10/hour, and reducing the batch or enabling gradient
    checkpointing would no longer be the unmodified starter.

## Training scale vs the paper

| | Paper (Search-R1, App. B.2) | Ours |
|---|---|---|
| Fine-tuning | Full model (actor, and a full critic for PPO) | Rank-16 LoRA (actor; plus critic LoRA and value head for PPO) |
| Hardware | 8 × H100 | 1 × L40S |
| Prompts per step | 512 | GRPO 8 (×5 samples = 40 rollouts); PPO 40 |
| Training steps | 500 (paper text; v0.2 scripts use 1,005) | in progress (≈40 per algorithm expected) |
| Rollouts in total | GRPO ≈ 1.28M, PPO ≈ 256k | in progress (≈1,600 per algorithm expected) |
| Training data | Full NQ + HotpotQA train merge (~170k) | 4,000-question frozen subset |

## Cost (Modal, shared class workspace)

| Stage | GPU | Approx. cost |
|---|---|---|
| Retrieval assets download and corpus repair | CPU | $0.06 |
| Data preparation | CPU | $0.01 |
| Direct Inference | L4 | $0.16 |
| Released GRPO + PPO evaluation (incl. one failed launch) | L40S + 96 GiB RAM | ≈ $2.20 |
| Starter 20-step run (two attempts) | A100-80GB | ≈ $0.60 |
| LoRA GRPO + PPO training | L40S + 96 GiB RAM | in progress (≈ $10 expected) |
| Retrieval volume storage (~74 GiB at $0.09/GiB-month) | — | ≈ $0.22/day |

Costs are computed from Modal's published rates (`modal billing rates`) and measured container
time.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # plus CPU torch for tests
python -m unittest discover -s tests                                             # 78 CPU tests
modal run --detach modal_retrieval.py                       # E5 index, corpus, encoder -> volume
modal run modal_retrieval.py --job index                    # corpus offsets
modal run modal_app.py --job prepare                        # Bamboogle + training subset + revision lock
MODAL_GPU=L4 modal run modal_app.py --job evaluate --method direct --run-name direct-v1 --acknowledge-compute
MODAL_GPU=L40S MODAL_MEMORY=98304 modal run modal_app.py --job evaluate --method released_ppo \
    --run-name released-ppo-v02b --retrieval-url local --acknowledge-compute
MODAL_GPU=L40S MODAL_MEMORY=98304 modal run --detach modal_app.py --job train --algorithm grpo \
    --profile short --run-name grpo-short --retrieval-url local --minutes 100 --acknowledge-compute
python scripts/summarize_results.py --runs downloaded-runs  # after: modal volume get ... /runs
```

## Repository layout

```text
search_r1_team/     model (frozen Qwen + LoRA actor/critic), GRPO/PPO objectives, multi-turn rollout, retrieval client
scripts/            data prep, training, inference, scoring, retrieval server/downloader, results summary
configs/            smoke, short and experiment configurations
modal_app.py        Modal runner: data prep, training, evaluation, co-located E5 retriever
modal_retrieval.py  CPU jobs: stream the E5 assets to a volume, build corpus offsets
modal_starter.py    separate environment for the class starter's own train.py
docs/               paper parity (with upstream citations), setup, protocol
results/            summary.json, training curves, selected traces
tests/              CPU unit tests (synthetic fixtures)
```
