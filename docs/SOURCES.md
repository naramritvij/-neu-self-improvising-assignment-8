# Source register

Course requirements come from the assignment screenshots we were given. They are not
replaced by classmates' proposed hyperparameters. See ASSIGNMENT_SCOPE.md.

## Class starter

https://github.com/suhabe/tiny-zero-lora
Pinned inspected commit: `3659acfad61dab2f90fa46294566ff6808452e87`.
The fetcher downloads upstream into ignored vendor storage; no unlicensed copy
of the class repository is bundled into this group's source tree.

## Paper and authors' implementation

- https://arxiv.org/html/2503.09516v5
- https://github.com/PeterGriffinJin/Search-R1
- https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py
- https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py
- https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

## Evaluation and training data

- Course sheet: https://docs.google.com/spreadsheets/d/1jwcsA5kE4TObr9YHn9Gc-wQHYjTbLhDGx6tmIzMhl_U/edit?gid=0
- https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
- https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus
- https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index

## Model and deployment documentation

- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/peft
- https://huggingface.co/docs/peft/index
- https://modal.com/docs/guide/workspaces
- https://modal.com/docs/guide/getting-started
- https://modal.com/docs/guide/volumes

Floating source links are references; actual model/data revisions are locked by
preparation scripts before running. Published scores are explicitly separate from
team measurements. No claim of exact paper parity follows merely from these links.
