# Modal setup — start with the class workspace

No Modal job has been launched during repository preparation. Workspace access,
quota and budget are not verified. Do not put invitation links or tokens in Git.

## 1. Authenticate locally

```bash
python -m pip install modal
modal setup
modal profile list
modal profile activate CLASS_WORKSPACE_PROFILE
```

Replace the profile with the actual class-workspace profile. Authenticate through
your own browser. Website workspace selection does not by itself establish the
CLI profile. Do not paste credentials into chat. First stop here and verify the
class workspace and permitted GPUs; do not immediately run the commands below.

## 2. Reproduce the class starter in its own environment

```bash
python scripts/fetch_starter.py
modal run modal_starter.py --run-name starter-smoke --acknowledge-compute
```

The image follows the inspected class dependencies and requests B300 by default.
Availability and compatibility are not verified. `MODAL_STARTER_GPU` can override
the GPU. After a successful smoke check and an approved budget:

```bash
modal run modal_starter.py --run-name starter-full --full --steps 2000 --acknowledge-compute
```

Outputs go to `/vol/starter/`. Record real logs. The compact implementation uses
a separate environment; do not mix the two dependency stacks.

## 3. Prepare data

```bash
modal run modal_app.py --job prepare
```

This is a CPU cloud job and can still incur charges. It prepares Bamboogle and a
frozen NQ/HotpotQA train/validation subset under `/vol/data`, with revision locks.
If the sheet export needs login, export CSV in your browser and use the local
`prepare_bamboogle.py --csv PATH` route, then upload prepared data to the volume.
Inspect partially prepared folders rather than silently overwriting evidence.

## 4. Configure real retrieval

Search training and inference require a reachable E5 retrieval API. Laptop
localhost is not the localhost of a Modal GPU container. See RETRIEVAL_SETUP.md.
Record the actual corpus and encoder revision. For an authenticated endpoint,
create a Modal secret named `search-r1-retrieval` containing `RETRIEVAL_API_KEY`
and set local `USE_RETRIEVAL_SECRET=1`. Never commit that secret.

## 5. GPU smoke runs

The compact runner requests A100-80GB by default; `MODAL_GPU` overrides it.
Replace every capitalized placeholder below with verified values:

```bash
modal run modal_app.py --job train --algorithm grpo --profile smoke --run-name smoke-grpo --retrieval-url RETRIEVAL_URL --corpus-id CORPUS_REVISION --retriever-id E5_REVISION --acknowledge-compute
modal run modal_app.py --job train --algorithm ppo --profile smoke --run-name smoke-ppo --retrieval-url RETRIEVAL_URL --corpus-id CORPUS_REVISION --retriever-id E5_REVISION --acknowledge-compute
```

Check finite losses, successful search, actor/critic updates and saved checkpoints.
PPO's first smoke step warms up the critic. Smoke results are not final experiments.

## 6. Approved experiments and evaluation

Only after successful smoke checks and scope/budget approval, switch to
`--profile experiment` with distinct run names `grpo-main` and `ppo-main`.
The profiles propose 500 steps and 4 prompts × 8 rollouts; this is not an approved
minimum or an estimate of runtime/cost. `--steps N` records a changed budget.

```bash
modal run modal_app.py --job evaluate --method direct --run-name direct --acknowledge-compute
```

Own-adapter evaluation uses `--method own_grpo` or `own_ppo`, the corresponding
run name, `--checkpoint /vol/runs/RUN/checkpoints/step_NNNNN`, and the same verified
retrieval arguments. Use the actual path from latest_checkpoint.json.

Released-checkpoint inference instead uses `--method released_grpo` or
`released_ppo` with retrieval arguments and no adapter checkpoint. Keep this track
separate. `--limit N` is only a diagnostic and is rejected by the final report.

## Resume, download and report

Training resume uses the same run name plus `--resume --checkpoint PATH` for the
latest complete checkpoint. Data/retrieval/configuration must match; only total
steps may be extended. Load optimizer state only from your own trusted files.
Inference can use `--resume` with unchanged settings.

```bash
modal volume get info7375-search-r1-team /runs ./downloaded-runs
```

Place evaluation folders under local `runs/evaluation/` and training folders under
`runs/`, then use scripts/build_report.py. Retain manifests, predictions and real
logs. Do not commit large checkpoints or corpora. Each complete training checkpoint
is explicitly committed to the volume; a hard kill may still lose uncommitted
writes. Cloud persistence must be checked in an actual run.

References:
- https://modal.com/docs/guide/workspaces
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/getting-started
