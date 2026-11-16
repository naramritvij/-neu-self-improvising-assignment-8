# Validation record — 2026-11-16

## Checks actually executed locally

```bash
python -m unittest discover -s tests -v
python -m compileall -q search_r1_team scripts tests modal_app.py modal_starter.py
```

**Result:** 67 tests passed; Python compilation passed. The local runtime used
PyTorch `2.10.0+cpu`; CUDA was unavailable. The workflow requests a separate
PyTorch 2.6 CPU environment. Do not equate local checks with a verified Actions run.

Coverage includes group advantages, GAE, clipped actor/value objectives, synthetic
actor/critic updates, frozen base/reference behavior, retrieved-token masking,
data separation, answer extraction/normalization, evaluation coverage and hash
checks, corpus indexing, and direct-model adapter activation.

## Fixes during repository preparation

- Plain Transformers models can expose `set_adapter` even with no adapter loaded.
  HFPolicy now checks whether it actually has a PEFT wrapper before selecting an
  adapter. Four regression tests cover direct, actor, critic and no-grad modes.
- Modal uploads only the source/config directories rather than the entire local
  project folder, preventing unrelated root credentials from being swept in.

Reference for adapter API behavior:
https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/peft

## What these checks do not prove

They use synthetic CPU fixtures, not Qwen2.5-3B weights. They do not establish
GPU dependency compatibility, convergence, paper parity, cloud persistence,
full-index retrieval behavior or model quality. Neither Transformers nor PEFT
was installed for these local fixture tests.

## Execution still pending

The exact class starter, actual GPU image builds, Qwen integration, real data
preparation, retrieval deployment, GRPO/PPO training, full Bamboogle inference,
measured scores and result figures. No Modal job was launched as part of this
repository preparation. No synthetic fixture is submitted as experiment evidence.
