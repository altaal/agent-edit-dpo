---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags: [dpo, lora, coding-agent]
---

# Agent Edit DPO adapter

A small experimental LoRA adapter trained by Ali Taalimi for the public
[Agent Edit DPO experiment](https://github.com/altaal/agent-edit-dpo).

Base: Qwen/Qwen2.5-0.5B-Instruct, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. This directory contains only the
trained adapter, not the base weights. The adapter uses Apache 2.0, matching
the base model license; see LICENSE in this directory.

Training: 40 synthetic preference pairs from five authored Python repair tasks,
TRL DPO beta 0.1, 40 steps, seed 7, float32, rank-8 LoRA on q_proj/v_proj.
Only 540,672 adapter parameters were trainable; base weights stayed frozen.
The chosen edits were test-verified correct solutions, while rejected edits
returned constants. These are easy negatives, not human preferences or rollouts.

Use the repository's download and evaluation commands to load the exact base
revision and compare adapter-disabled versus adapter-enabled agent runs.
[Metrics](../metrics.json) and [configuration](../config.json) describe the actual
training. [Before/after results](../../../results/week6-before-after/summary.md)
and [failure analysis](../../../FAILURES.md) describe the observed limits.

This is a small public pipeline demonstration. It is not a general coding model,
not a production agent, and not evidence of improved capability without reading
the paired evaluation. Evaluation tasks are public authored fixtures. Repeated
greedy runs do not create independent task samples.
