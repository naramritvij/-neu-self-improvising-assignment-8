# Search-R1 replication — experiment report

**Team:** Ritvij and Ghouse · **Course:** INFO 7375 · **Assignment:** 8

**Status: NOT SUBMISSION-READY.** Implementation is prepared; real experiments
have not run. No conclusion about model quality or GRPO versus PPO is supported.

| Experiment | Measured exact match | Correct / evaluated | Status |
|---|---:|---:|---|
| Direct inference | — | — | Not run |
| Own LoRA + GRPO | — | — | Not run |
| Own LoRA + PPO | — | — | Not run |
| Released GRPO checkpoint | — | — | Not run |
| Released PPO checkpoint | — | — | Not run |

Published reference scores are in paper_reference.json and are not our results.
No placeholder chart is presented as a measured output.

## Planned method

Frozen Qwen2.5-3B-Instruct base, rank-16 LoRA, independent GRPO/PPO tracks, a
separate PPO critic adapter/value head, real multi-turn retrieval, observation-
token loss masking and held-out Bamboogle exact-match evaluation. Released-model
inference is tracked separately from new LoRA training.

## Evidence still required

Starter reproduction log; verified GPU images and Qwen smoke runs; data/model/
retriever provenance; agreed training budget; actual checkpoints and logs; complete
held-out predictions; measured figures; error analysis and truthful contributions.

## Limitations to discuss

The compact training framework, prompts, subset, LoRA architecture, truncation
rules and budget differ from paper-scale runs. See ../docs/PAPER_PARITY.md.
Resolve the inference-only scope before deciding which training runs are required.
