# Experiment protocol

## Separate experiment tracks

Direct inference uses unmodified `Qwen/Qwen2.5-3B-Instruct`, without retrieval
or RL training. It is one shared baseline. Own GRPO and own PPO start independently
from the same frozen model revision with LoRA. Released-checkpoint inference is
recorded separately and is not evidence of training a new adapter.

## Data integrity

The inspected course sheet uses Question, Answer and Answer 2 headers. Preserve
answer aliases and exclude unrelated metadata. Store questions separately from
evaluator-only labels and record source/processed hashes. Never pass Bamboogle
labels into inference, training, prompt tuning or checkpoint selection.

Use the same frozen NQ/HotpotQA training subset and independent validation set for
both algorithms. The preparer removes duplicate and overlapping held-out questions.
Freeze model and data revisions in sources.lock.json before training.

## Fair comparison

Hold model revision, LoRA modules, prompt, data and retrieval settings fixed.
Record actual prompts, rollouts, generated tokens, search calls and runtime:
equal optimizer steps do not mean equal compute. Use identical evaluation rules.
The proposed 500-step, 4-prompt, 8-rollout configuration is not an approved minimum.

Implement generate → search → insert observation → continue → answer. Retrieved
information stays in context but is excluded from policy/value action accounting.
Bound tokens, search calls and timeouts; log passages and termination reasons.
A changed corpus, retriever, prompt or live web search is a disclosed deviation.

## Scoring

Normalized exact match lowercases, removes ASCII punctuation/articles, collapses
whitespace and compares accepted aliases. It is not substring matching. Search
outputs require a complete answer tag; the scorer takes the last one in response-
only output. Direct output is treated as a bare answer. See PAPER_PARITY.md for
upstream parsing differences.

Reject unknown, duplicate and missing prediction IDs. Keep failed attempts as
empty predictions so they count as incorrect. Partial diagnostics and smoke-only
adapters do not belong in the final comparison.

## Evidence to retain

For each real run save the git revision, environment, GPU, dataset hashes, base
revision, adapter settings, retrieval provenance, seed, budgets, complete logs,
checkpoint and predictions. Use the fixed-budget checkpoint or independent
validation for selection, not Bamboogle. One seed does not establish a robust
algorithm ranking. We review the final report before submission.
