# Ritvij and Ghouse — 50/50 work plan

**INFO 7375 · Assignment 8** · Two-person team approval is pending.

Equal substantive work is the goal, not equal commit counts. Effort points are
planning estimates, not claims of completed work. PPO receives more training
points because of the critic; Ritvij balances that with integration/reporting.

| Workstream | Ritvij | Points | Ghouse | Points |
|---|---|---:|---|---:|
| Starter | Set up and reproduce the class example; record environment and outputs | 6 | Independently verify it; explain rewards, LoRA and training flow | 6 |
| Foundations | Prepare Bamboogle, verify scorer, run direct baseline | 12 | Retrieval, multi-turn search, traces and observation masking | 12 |
| RL experiment | Implement, debug and run LoRA + GRPO | 20 | Implement, debug and run LoRA + PPO, including critic/value learning | 24 |
| Evaluation/report | GRPO evaluation, comparison plots, methodology and integration | 12 | PPO evaluation, error analysis, PPO details and final review | 8 |
| **Total** | **50%** | **50** | **50%** | **50** |

## Task acceptance

| Task | Owner | Completion evidence |
|---|---|---|
| R1: starter run | Ritvij | Pinned commit, real log, environment and saved output |
| G1: starter verification | Ghouse | Reproduction/verification notes on rewards, prompts and adapters |
| R2: baseline/evaluation | Ritvij | Frozen data hashes, complete baseline predictions and scorer checks |
| G2: retrieval | Ghouse | Reachable endpoint, source revisions, trace and masking checks |
| R3: GRPO | Ritvij | Successful smoke run, agreed experiment, adapter and measured logs |
| G3: PPO | Ghouse | Actor/critic smoke run, agreed experiment, adapter and measured logs |
| R4: integration | Ritvij | Combined metrics, actual plots and reproducibility section |
| G4: analysis | Ghouse | PPO evaluation, success/failure examples and report review |

These are task cards in this document, not a claim that GitHub issues exist.

## Working agreement

Both read the paper and understand both algorithms. Review the other person's
changes before merging. Transfer a documented task when actual effort becomes
uneven. Record genuine contributions in docs/CONTRIBUTIONS.md.

Use `main` for the shared baseline, `ritvij/grpo` and `ghouse/ppo` for separate
work, and `shared/...` branches for common fixes. Both tracks must use the same
frozen dataset, scoring contract and retrieval configuration.

## Milestones

Confirm scope/workspace → reproduce starter → baseline and retrieval smoke checks
→ GPU smoke tests → approved experiments → frozen evaluation → joint report review.

No calendar deadline is invented: the screenshot says “due in 1 week” without
establishing the actual date. AI-assisted initialization is not independent human
implementation or completed 50/50 work.
