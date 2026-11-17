# Project plan

**INFO 7375 · Assignment 8** · Authors: Ritvij Naram and Ghouse

## Workstreams

| Workstream | Deliverable | Status |
|---|---|---|
| Starter | Understand and replicate `suhabe/tiny-zero-lora` (walkthrough and its own `train.py`) | Done; see README |
| Data and evaluation | Frozen Bamboogle (FlashRAG) set, exact-match scorer, Direct Inference | Done |
| Retrieval | E5 / Wikipedia-2018 index, corpus and encoder on a Modal volume; in-container server | Done |
| Released checkpoints | Search-R1 v0.2 GRPO and PPO evaluated with search | Done |
| LoRA GRPO and PPO | Short training runs and Bamboogle evaluation of both adapters | Done |
| Report | Results table, traces, failure analysis, cost, scale comparison | Done |

## Working agreement

Both algorithms use the same frozen dataset, scoring contract and retrieval configuration. We
review changes before merging to `main`; feature work happens on branches.
