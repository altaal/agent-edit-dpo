# What failed / what I learned

The complete before/after counts are in the
[evaluation table](results/week6-before-after/summary.md). Every declared attempt
is retained, including invalid outputs and action-limit failures. The experiment
uses ten authored tasks, repeated with greedy decoding, not 30 independent tasks.

**Final result: base 0/30; DPO 0/30.** All 60 attempts exhausted six actions without
submitting, and none had a correct final patch. The result provides no evidence of
improved agent performance from this training run. A stronger experiment would
first establish reliable tool use on separate development tasks; these scores
cannot settle whether DPO improves an already competent tool-using model.

## 1. A plausible tool name is not a usable tool call

In [clamp, base, attempt 1](results/week6-before-after/clamp--base--1.json), the
model selected `edit` but requested lines 0 through 5 in a two-line function.
The tool rejected the range and kept the source unchanged. Subsequent responses
mixed unusable output with another invalid edit. The six-action limit ended the run.
The [DPO version](results/week6-before-after/clamp--dpo--1.json) made the same first
request and also failed to recover. Neither produced a correct final function.

The boundary checks did their job. The small model did not consistently follow
the editor's one-based, inclusive line contract. A completion that looks like JSON
is not enough; its arguments must be valid for the current file.

## 2. Changing the response is not the same as improving the task

[Safe mean, base](results/week6-before-after/safe-mean--base--1.json) produced 1,184
completion tokens across six calls; [safe mean, DPO](results/week6-before-after/safe-mean--dpo--1.json)
produced 295. Both began with an invalid line range and a placeholder replacement.
Neither submitted or fixed the function. Shorter output is not a success metric.

Training changed the adapter parameters and some generated text. That demonstrates
a working training-and-inference pipeline, not improved agent performance.

## 3. The preference data covers one edit, not an entire repair

[The dataset](data/preferences.jsonl) contains one initial prompt and two alternative
full-function edits. The chosen code passes tests; the rejected code returns a
constant. No preference contains a recovery conversation, a `test` action followed
by its observation, or a successful `submit` action.

The data therefore leaves much of the evaluated behavior uncovered. It also has
only five task families and easy negatives whose answers are shorter. This limits
what a 40-step update can establish. A lower training loss cannot stand in for the
separate tool-loop evaluation. Keep these limits explicit rather than explaining
away a poor final score.

## An implementation correction caught before the final run

The first one-step smoke run counted the chat-template return container rather
than its token IDs. That made its token-length diagnostic wrong. The final training
code renders the template to text and counts the encoded token IDs explicitly.
The final run measured a maximum of 667 tokens, below the fixed 1,024-token limit;
the trainer retained all 40 training rows. The full run started from fresh base
weights, not the smoke adapter. [Training provenance](training/main/provenance.json)
records this correction.

Lesson: check the actual library return type before trusting a diagnostic. Use the
measured sequence length and the trainer's retained row count together before
claiming that training examples were not truncated.
