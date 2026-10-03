# Agent Edit DPO

Can a tiny preference-training run improve a small model's tool-driven code repairs?

Start with the [Week 5–6 technical walkthrough](TECHNICAL_OVERVIEW.md): exact goals,
changes from earlier weeks, source excerpts, and the path from one preference pair
to training and the saved clamp evaluation files. The measured outcome was no
improvement; the walkthrough separates learning value from that result.

This repo connects **verified preference data → actual DPO training → an agent
loop → the same evaluation before and after training**. It reuses
[ACI Patch Agent](https://github.com/altaal/aci-patch-agent) at a pinned commit.
The [plain-language six-week guide](https://github.com/altaal/aci-patch-agent/blob/main/WEEK_BY_WEEK.md)
explains the weekly tasks, examples, budgets, and ship gates.
This repository covers Weeks 5–6. In the sibling-project workspace, the guide's
canonical editable source is `../WEEK_BY_WEEK.md`.

## What is included

- Fifty synthetic edit preferences, each checked in a Docker container.
- One actual 40-step DPO run on `Qwen/Qwen2.5-0.5B-Instruct` with a LoRA adapter.
- The 2.2 MB trained adapter, exact model revision, training configuration, and metrics.
- The same four-tool agent evaluated with the adapter disabled and enabled.
- Per-attempt traces, numerical results, and an honest account of failures.

Out of scope: GRPO, training a reward model, a model sweep, distributed training,
arbitrary repository repairs, or general coding-capability claims.

## Install and inspect without downloading a model

Requires Python 3.11+. The tested training environment used Python 3.12, a Mac with
32 GiB memory, and PyTorch's MPS GPU backend. The scripts fall back to CPU, which may
be much slower. Docker is required to execute generated Python and verify data.

```sh
git clone https://github.com/altaal/agent-edit-dpo.git
cd agent-edit-dpo
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m agent_edit_dpo.verify
python -m agent_edit_dpo.evaluate --report-only --output results/week6-before-after
```

These commands use no model API key. The report regenerates the committed scores
from traces; it does not run a model or independently re-grade patches.
The verifier checks saved labels, task separation, training/data/weight hashes,
source hashes, all 60 attempt identities, and the reported denominators. These are
artifact-consistency checks, not a new training or behavioral test run.

## Run the before/after comparison

Install the pinned training dependencies and download the exact public base model.
The base weights are about 1 GB and are not copied into Git. The committed adapter
is loaded on top of those weights.

```sh
python -m pip install -e '.[train]'
python -m agent_edit_dpo.download
docker pull python@sha256:da047cb8f9d1d98e5c070f5300ba9f7274e33b8fc0e5be5ed88740aed1b95ba9
python -m agent_edit_dpo.evaluate --output runs/before-after
```

The last command runs ten tasks × two conditions × three repetitions, saving 60
attempts in a new directory. It makes local model calls and no paid API calls.
Existing output directories are refused.

## What the training data means

The five development tasks are normalize-method, chunked, merge-intervals,
parse-bool, and unique-stable. For each task, an authored correct implementation
is paired with ten distinct constant-return mistakes. Both alternatives use the
same JSON `edit` format and replace the entire original function. The preference
label comes from final behavioral checks: the chosen edit passes all checks and
the rejected edit fails at least one. Syntax validity alone is not the label.

[All 50 pairs](data/preferences.jsonl), [verification records](data/verification.json),
and [provenance and task split](data/manifest.json) are committed.

Forty pairs train the adapter. Ten are reserved diagnostic pairs and were not used
for training or model selection. They share prompts and chosen answers with the
training pairs, so they are **not** a held-out generalization set. The independent
check is the ten different evaluation functions from ACI Patch Agent. Their IDs
are disjoint from all five training tasks. These are public authored fixtures,
not a contamination-resistant benchmark.

This is synthetic data from **five task families**, not 50 independent tasks, human
preferences, or agent-collected rollouts. The rejected answers are easy negatives,
and chosen answers are systematically longer. Those limits matter when interpreting
the result.

Rebuild and re-verify the data in a separate directory:

```sh
python -m agent_edit_dpo.data --output runs/rebuilt-data
```

## What actually trained

| Setting or measurement | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Base revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Objective | TRL DPO, sigmoid preference loss, beta 0.1 |
| Adapter | LoRA, rank 8, alpha 16, q_proj and v_proj, dropout 0 |
| Trainable parameters | 540,672; base weights frozen |
| Data and updates | 40 pairs, batch 1, one epoch, 40 optimizer steps |
| Learning rate | 0.00001 initially, linear decay |
| Seed and precision | 7; float32 |
| Longest training sequence | 667 tokens, below the 1,024-token limit |
| Mean training loss | 0.63836 |
| Reference scoring, training, and adapter save | 111.39 seconds on the recorded MPS machine; excludes model download/load |
| Changed trainable tensors | 96 |

[Training metrics](training/main/metrics.json) include per-step logs;
[configuration](training/main/config.json) pins library versions and data hash;
[serialized trainer settings](training/main/trainer_configuration.json) record the
remaining options. The [adapter](training/main/adapter/) is committed. Training loss
and a nonzero parameter update prove optimization ran; they do not prove task improvement.

Reproduce one fixed training run using the committed dataset:

```sh
python -m agent_edit_dpo.train --output runs/retrained --steps 40
python -m agent_edit_dpo.evaluate --adapter runs/retrained/adapter --output runs/retrained-eval
```

The pipeline does not select a checkpoint using evaluation-task results. A one-step
smoke run checked that training worked before the final run, which started from a
fresh base model. No hyperparameter sweep was performed.

## Evaluation protocol

Both conditions use the same frozen base model, tokenizer, system instructions,
JSON tool protocol, ten functions, tests, six-action budget, and 384 generated-token
limit per call. `base` disables the trained LoRA adapter; `dpo` enables it. Evaluation
runs sequentially. Both use greedy decoding, so repeated attempts can be identical.
The 30 attempts per condition represent ten tasks, not 30 independent tasks.

The model must produce one complete JSON object selecting `view`, `edit`, `test`,
or `submit`. Invalid text consumes an action and receives feedback. The tools use
the original checked editor and isolated Docker evaluator. Passing requires an
explicit submission and all final checks. API/runner errors, invalid output, and
exhausted action budgets remain failures in the declared matrix.

The base and DPO conditions are comparable **to each other**. Their scores should
not be compared as a training gain against the larger hosted model in the other
repos: that model, its native tool-call interface, and its action limit differ.

Measured scores and concrete failure examples are in [the saved table](results/week6-before-after/summary.md)
and [what failed / what I learned](FAILURES.md).

## Measured outcome: no improvement

September 28, 2026, all 60 declared attempts completed:

| Condition | Submitted and passed | Correct final patches | Mean actions |
| --- | ---: | ---: | ---: |
| Base model | 0/30 (0%) | 0/30 | 6 |
| DPO adapter | 0/30 (0%) | 0/30 | 6 |

All attempts exhausted the six-action budget without submitting. None produced a
correct final patch. Training changed the weights and some outputs, but this setup
did **not** improve completed repairs. Invalid tool arguments and unusable response
formats dominated the observed failures. This establishes a runnable training and
evaluation pipeline, not a capability gain or a general conclusion about DPO.

The shared report format also contains recovery counters (`fault_exposed`,
`passed_when_exposed`, `failed_actions`, `repeated_actions`). This experiment does
not collect those counters; their zero defaults are **not measurements** of tool
failures. Use the saved events and the primary pass/action metrics above.

## Sources and licenses

Training uses the implementation in [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer).
The objective is described in the [DPO paper](https://arxiv.org/abs/2305.18290),
Section 4, Equation 7. This is an application of that implementation, not a new
training algorithm.

Code and authored fixtures use the repository's MIT license. The pretrained model
is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), whose
weights use Apache 2.0. The trained adapter is distributed under Apache 2.0; see
its model card and license. No employer data or private source is included.
