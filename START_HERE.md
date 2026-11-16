# Start here — Assignment 8 group project

**Repository:** `naramritvij/-neu-self-improvising-assignment-8`  
**Team:** Ritvij and Ghouse; planned work split 50/50.

The leading hyphen matches the repository Ritvij created. Do not create another
repository or upload an older ZIP.

## Ready versus pending

The repository contains source code, CPU tests, configurations, the team plan
and guarded Modal launchers. See README.md and VALIDATION.md. This is prepared
code, not completed GPU experiments or benchmark results.

## Your next account steps

1. Invite Ghouse through this repository's Settings → Collaborators. His GitHub
   username is not known here; the assistant has not sent an invitation.
2. Join/select the professor's shared Modal workspace. The screenshot of the
   personal `naramritvij` Starter workspace did not establish class access.
3. Verify two-person approval and the inference-only/training scope with the TA
   before substantial GPU spending. See docs/ASSIGNMENT_SCOPE.md.

Do not paste tokens into chat. Do not add a payment card just to follow these
instructions. Browser login and class-workspace selection are account steps
that you perform yourself.

## First technical checks

```bash
git clone https://github.com/naramritvij/-neu-self-improvising-assignment-8.git assignment-8
cd assignment-8
python scripts/preflight.py
```

With CPU PyTorch installed, tests require no model/dataset download:

```bash
python -m unittest discover -s tests -v
```

Then follow docs/MODAL_SETUP.md after confirming the class workspace. Do not
start with the 500-step profiles. First validate the starter, data and retrieval,
then small GPU smoke runs. No Modal job runs on GitHub push.

## Not yet done

Class-starter reproduction, Qwen-3B GPU integration, full corpus/index integration,
GRPO/PPO training, final Bamboogle predictions, plots and measured results remain
pending. The current report deliberately contains no invented scores.
