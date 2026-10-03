# Agent Edit DPO: goals and exact code flow for Weeks 5 and 6

This is a small engineering exercise in building a preference-training pipeline
and checking its effect on a coding agent. It applies an existing training method.
Its main value is connecting verified data, actual parameter updates, saved weights,
and a controlled evaluation that can report failure honestly.

**The saved outcome is no improvement: the base model passed 0/30 repairs and the
trained adapter passed 0/30.** Training changed parameters and some outputs; neither
condition produced a correct final patch. Those facts complete an experiment but
do not establish a useful repair model. [Recorded results](results/week6-before-after/summary.md).

| Week | Exact goal | Main artifact | What completion establishes |
| --- | --- | --- | --- |
| 5 | Verify preferences and train one adapter on 40 selected pairs. | Data proofs, training configuration/metrics, saved adapter. | Optimization ran and parameters changed. |
| 6 | Measure whether that adapter improves complete repairs on ten different tasks. | 60 agent traces and before/after table. | This particular training run did not improve completed repairs under this evaluation. |

## Week 5: goal and difference from Week 4

**Goal:** create examples of preferred and rejected edits, verify them with tests,
and run one real training job. DPO means direct preference optimization: here,
the trainer updates a model using a preferred response and a rejected response
to the same prompt. A LoRA adapter is the small collection of trainable parameters
saved separately from the pretrained model's frozen base weights.

**Why do it?** Learn the complete data-to-weights process, including how to check
that the labels were verified, the intended rows were selected, optimization ran,
and a later evaluator loads the same adapter. There is no new training algorithm.
The open empirical question, tested in Week 6, is whether this very small update
helps the model complete tool-driven repairs.

| Aspect | Week 4: recovery comparison | Week 5: preference training |
| --- | --- | --- |
| What changes | Text accompanying a tool fault | Adapter parameters |
| Model | Hosted `qwen/qwen3-next-80b-a3b-instruct` | Local `Qwen/Qwen2.5-0.5B-Instruct` |
| Data/tasks | Ten evaluation tasks in a live loop | Five Week 1 development tasks; 50 authored preference pairs |
| Fault injection | One timeout or truncated response per exposed attempt | None |
| Model interface | Hosted native tool calls | A text JSON tool format used in both training and later evaluation |
| Main measurement | Completed repairs and fault exposure | Verified labels, training steps, changed parameters, saved weights |

This is a change of model and experimental question. Week 6's score cannot be
compared with Week 4's score as an effect of DPO: model size, interface, fault
injection, source visibility, and action budgets differ.

## Week 6: goal and difference from Week 5

**Goal:** determine whether the saved Week 5 adapter improves completed repairs
compared with the same small model with the adapter disabled. Training loss alone
does not answer that question.

Week 5 updates and saves parameters. Week 6 freezes the resulting model for inference,
loads the adapter, and uses `base` and `dpo` as two conditions. Both receive the same
ten evaluation tasks, instructions, checked editor, JSON protocol, six-action budget,
and 384-token generation limit per call. Both start with the source in the prompt.
Neither uses `FaultWorkspace` from the recovery repository.

There are three fresh attempts for each task and condition, or 60 records. Decoding
is greedy (`do_sample=False`), so repeated attempts can be identical; these are ten
task definitions, not thirty independent tasks per condition.

The agent must make valid tool calls, repair the function, and submit it. Final
grading runs even if no submission occurs, but success requires both submission
and passing all final checks. This tests a full sequence of actions, although the
training dataset only teaches a single initial edit. That mismatch is one limit
of the exercise; the measured result does not prove it is the sole cause of failure.

## Directory and source versions

```text
agent-edit-dpo/
  agent_edit_dpo/
    download.py     Download the recorded base-model revision into .model/
    goldens.py      Authored fixes for five development functions
    data.py         Pair fixes with constant-return mistakes; verify in Docker
    protocol.py     Build text prompts and translate JSON responses to tool calls
    train.py        Select 40 pairs; run DPO; save adapter and measurements
    evaluate.py     Load model/adapter; run base and dpo agent attempts
    verify.py       Check consistency of the committed evidence
  data/            preferences.jsonl, verification.json, manifest.json
  model-source.json              Exact base-model identity and revision
  .model/                        Downloaded base weights; ignored by Git
  training/main/
    adapter/                     Saved LoRA configuration and weights
    config.json                  Data hash, model, versions, and training settings
    trainer_configuration.json  Serialized trainer options
    metrics.json                 Steps, loss, parameter changes, and per-step logs
    provenance.json              Record of the published training run
  results/week6-before-after/    60 attempts, manifest, CSV, and summaries
  tests/test_protocol.py         Tests using authored data and parser inputs
  runs/                         New local outputs, ignored by Git
```

The repository's source inspected for this guide is commit `bf3c426`.
[pyproject.toml](pyproject.toml) pins the reused agent to
`42a2683b643b3b9af2f034529032444accf0c849`. That dependency supplies `Task`, `Workspace`,
`run_task`, `DockerSandbox`, and `execute_matrix`; the neighboring ACI checkout can
contain later code and is not automatically imported. The pinned loop has no later
submission-review or conversation-restart logic. A valid submit ends that loop.

The saved evaluation manifest records hashes of both the dependency and DPO source.
The local DPO sources match its recorded hashes. `python -m agent_edit_dpo.verify`
checks DPO source/data/adapter consistency; this is not a live behavioral rerun.

## Commands and what they do

Run commands from this repository's root with its Python environment active.
Follow [README installation](README.md#install-and-inspect-without-downloading-a-model)
and [model setup](README.md#run-the-beforeafter-comparison) for dependencies, the
model download, and the pinned Docker image. New output directories must not exist.

| Command | Work performed | Model/training or Docker execution? |
| --- | --- | --- |
| `python -m unittest discover -s tests -v` | Check the protocol and candidate construction | No model or Docker |
| `python -m agent_edit_dpo.verify` | Check committed records, hashes, identities, and scores | No model or Docker |
| `python -m agent_edit_dpo.evaluate --report-only --output results/week6-before-after` | Rewrite reports from saved attempts | No model or Docker |
| `python -m agent_edit_dpo.download` | Download the pinned base model into `.model/` | Network download, no training |
| `python -m agent_edit_dpo.data --output runs/week5-data` | Verify both alternatives for all 50 preference pairs | Docker; no model |
| `python -m agent_edit_dpo.train --output runs/week5-training --steps 40` | Read committed `data/preferences.jsonl`, update adapter, save it | Local training; no Docker or paid API |
| `python -m agent_edit_dpo.evaluate --adapter runs/week5-training/adapter --output runs/week6-eval` | Compare the newly trained adapter with its base | Local model inference and Docker; no paid API |

The data and training commands are separate entry points. **Writing
`runs/week5-data` does not make training use that directory.** `train.py` currently
reads `data/preferences.jsonl` directly and has no `--data` option. Rebuilding into
`runs/` verifies the data-generation path while preserving the committed dataset.
Likewise, evaluation defaults to `training/main/adapter`; supply `--adapter` as
above to evaluate a newly trained adapter.

## Week 5 code flow: authored functions to verified preference pairs

```text
python -m agent_edit_dpo.data --output runs/week5-data
  data.main()
    create new directory; DockerSandbox.preflight()
    verify development-task IDs and evaluation-task IDs are disjoint
    candidates()
      for each of five TASKS, pair it with its authored SOLUTIONS entry
      for each of ten constants, build a wrong constant-return function
      build prompt + chosen edit + rejected edit; label train or diagnostic
    sandbox.check(task, chosen, final=True)
    sandbox.check(task, rejected, final=True)
    require chosen passes, rejected fails, and neither check has a runner error
    write preferences.jsonl + verification.json + manifest.json
```

The loop creates ten rows from each task. The first eight per task are training
rows; the remaining two are diagnostic rows. That produces 40 training rows and
10 diagnostic rows from only five task families:

Source: [agent_edit_dpo/data.py](agent_edit_dpo/data.py#L17), lines 17–28. Exact excerpt:

```python
CONSTANTS = ("None", "0", "1", "True", "False", "[]", "{}", "''", "'GET'", "'bad'")


def candidates():
    for task, solution in zip(TASKS, SOLUTIONS, strict=True):
        for index, constant in enumerate(CONSTANTS):
            rejected = task.source.splitlines()[0] + f"\n    return {constant}\n"
            yield task, solution, rejected, {"id": f"{task.id}-{index:02}", "task": task.id,
                "split": "train" if index < 8 else "diagnostic",
                "prompt": text_messages(prompt_for(task)),
                "chosen": [{"role": "assistant", "content": edit_response(task, solution)}],
                "rejected": [{"role": "assistant", "content": edit_response(task, rejected)}]}
```

`goldens.py` contains authored correct code. A model does not propose or grade these
pairs. Docker checks determine whether each authored label is supported:

Source: [agent_edit_dpo/data.py](agent_edit_dpo/data.py#L42), lines 42–52. Exact excerpt:

```python
    for task, chosen, rejected, row in candidates():
        chosen_check = sandbox.check(task, chosen, final=True)
        rejected_check = sandbox.check(task, rejected, final=True)
        assert chosen_check["passed"] and not rejected_check["passed"], row["id"]
        assert not chosen_check.get("error") and not rejected_check.get("error"), row["id"]
        rows.append(row)
        proofs.append({"id": row["id"], "chosen": chosen_check, "rejected": rejected_check})
        print(f"Verified {len(rows)}/50: {row['id']}", flush=True)
    encoded = "".join(json.dumps(row) + "\n" for row in rows)
    (output / "preferences.jsonl").write_text(encoded)
    (output / "verification.json").write_text(json.dumps(proofs, indent=2) + "\n")
```

### One exact pair: normalize-method-00

The task starts with `return str(method)`, which mishandles bytes. The chosen
function decodes bytes using ASCII, preserves strings, and rejects other types.
The rejected alternative returns `None`. Both are serialized as a full-function
`edit` covering original lines 1–2. These are decoded values from the first saved
[preference row](data/preferences.jsonl), not new model outputs:

**Chosen response:**

```json
{
  "tool": "edit",
  "arguments": {
    "start": 1,
    "end": 2,
    "replacement": "def normalize_method(method):\n    if isinstance(method, bytes):\n        return method.decode('ascii')\n    if isinstance(method, str):\n        return method\n    raise TypeError()\n"
  }
}
```

**Rejected response:**

```json
{
  "tool": "edit",
  "arguments": {
    "start": 1,
    "end": 2,
    "replacement": "def normalize_method(method):\n    return None\n"
  }
}
```


The saved row's `id` is `normalize-method-00`, `task` is `normalize-method`, and
`split` is `train`. Its `prompt` holds a system message with the JSON protocol and
a user message with the issue and original source. `chosen` and `rejected` each
hold one assistant response. [Verification records](data/verification.json) store
the independent checks for both alternatives under the same ID.

The diagnostic rows share the same task prompts and chosen fixes as training rows.
They are not a held-out generalization test. None of these preferences includes
`view → edit → test → submit`, a recovery dialogue, or any successful submission.
The independent task-ID split is between these five training tasks and the ten
evaluation tasks, which include clamp. All fixtures are public.

## Week 5 code flow: verified pairs to a saved adapter

```text
python -m agent_edit_dpo.train --output runs/week5-training --steps 40
  train.main()
    create output; seed 7
    read data/preferences.jsonl; select split == "train" → 40 rows
    read model-source.json; write config.json with data hash and versions
    load tokenizer + base model from .model/
    DPOConfig + LoraConfig → DPOTrainer
    check rendered sequence lengths; snapshot trainable parameters
    trainer.train() → 40 optimizer steps
    compare trainable parameters with snapshots; require a nonzero change
    save adapter/; attach model ID/revision to adapter configuration
    save metrics.json and trainer_configuration.json
```

The training-row selection is explicit:

Source: [agent_edit_dpo/train.py](agent_edit_dpo/train.py#L25), lines 25–26. Exact excerpt:

```python
    rows = [json.loads(line) for line in Path("data/preferences.jsonl").read_text().splitlines()]
    dataset = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows if r["split"] == "train"])
```

This passes the prompt and two alternatives to TRL's existing implementation;
the repository does not implement the DPO loss itself. The fixed settings and
adapter construction are:

Source: [agent_edit_dpo/train.py](agent_edit_dpo/train.py#L43), lines 43–53. Exact excerpt:

```python
    config = DPOConfig(output_dir=str(output), max_steps=args.steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-5, beta=0.1, max_length=1024,
        bf16=False, fp16=False, gradient_checkpointing=True, optim="adamw_torch",
        precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        save_strategy="no", logging_steps=1, report_to=[], seed=7, data_seed=7,
        dataloader_pin_memory=False, disable_tqdm=True)
    (output / "trainer_configuration.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    started = time.monotonic()
    trainer = DPOTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.0, task_type="CAUSAL_LM"))
```

`r=8` is the adapter rank; `q_proj` and `v_proj` are the model's projection modules
targeted by this configuration. Training updates the adapter while base weights
remain frozen. Both training and later evaluation use the same JSON response format.

The code verifies parameter change and saves the adapter here:

Source: [agent_edit_dpo/train.py](agent_edit_dpo/train.py#L62), lines 62–77. Exact excerpt:

```python
    before = {name: param.detach().cpu().clone() for name, param in trainer.model.named_parameters() if param.requires_grad}
    trained = trainer.train()
    changed = {name: float((param.detach().cpu() - before[name]).abs().max())
               for name, param in trainer.model.named_parameters() if name in before}
    assert any(value > 0 for value in changed.values()), "No adapter parameters changed"
    trainer.model.save_pretrained(output / "adapter")
    adapter_config = output / "adapter/adapter_config.json"
    adapter = json.loads(adapter_config.read_text())
    adapter.update(base_model_name_or_path=source["model_id"], revision=source["revision"])
    adapter_config.write_text(json.dumps(adapter, indent=2) + "\n")
    metrics = {**trained.metrics, "elapsed_seconds": round(time.monotonic() - started, 2),
               "global_step": trainer.state.global_step, "longest_training_sequence": max(lengths),
               "trainable_parameters": sum(p.numel() for p in before.values()),
               "changed_parameter_tensors": sum(v > 0 for v in changed.values()),
               "maximum_parameter_change": max(changed.values()), "log_history": trainer.state.log_history}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
```

The published run in [training/main/metrics.json](training/main/metrics.json) records
40 steps, 540,672 trainable parameters, 96 changed parameter tensors, and mean loss
0.63836. The longest sequence was 667 tokens under the configured 1,024-token limit.
Those observations establish that this optimization run executed; performance on
new repair tasks remains the Week 6 question.

## Week 6 code flow: load one model and switch the adapter

```text
python -m agent_edit_dpo.evaluate --output runs/week6-published-adapter
  evaluate.main()
    if --report-only: summarize saved files and return before model loading
    load base tokenizer + weights from .model/
    PeftModel.from_pretrained(base, training/main/adapter)
    move to MPS if available, otherwise CPU; set inference mode
    DockerSandbox.preflight()
    execute_matrix(output, EVAL_TASKS, [base, dpo], 3, runner, ..., workers=1)
      write manifest.json declaring 60 attempts
      for each attempt, sequentially:
        LocalClient(model, tokenizer, condition)
        imported run_task(task, client, sandbox, max_actions=6)
          fresh source + fresh conversation
          client.complete(messages) → local text generation → parsed tool call
          Workspace.execute → observation → next model call
          stop on submit, budget, or error; grade final source in Docker
        add identity and write <task>--<condition>--<repeat>.json
      write results.csv + summary.json + summary.md
```

The model and tokenizer are reused, but each attempt gets a new client, workspace,
and conversation. There is no training during evaluation and no repaired source
carried from one attempt to the next. One worker avoids concurrent adapter switching.

The adapter is loaded before running either condition:

Source: [agent_edit_dpo/evaluate.py](agent_edit_dpo/evaluate.py#L61), lines 61–64. Exact excerpt:

```python
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    sandbox = DockerSandbox()
    image = sandbox.preflight()
```

The exact matrix and loop call are:

Source: [agent_edit_dpo/evaluate.py](agent_edit_dpo/evaluate.py#L71), lines 71–73. Exact excerpt:

```python
    print(execute_matrix(args.output, EVAL_TASKS, ["base", "dpo"], 3,
        lambda task, condition, repeat: run_task(task, LocalClient(model, tokenizer, condition), sandbox,
                                                max_actions=6), metadata, workers=1))
```

### Inside LocalClient.complete: messages to one tool action

The reused agent calls `client.complete(messages)` regardless of whether its client
is remote or local. Here `LocalClient` supplies that method with local generation.
The client first converts tool observations into user messages prefixed with
`Tool result:` and converts previous tool calls into JSON text. It appends the
same `PROTOCOL` instructions used to construct the training prompts.
[Conversion code](agent_edit_dpo/protocol.py#L17).

After formatting/tokenizing those messages, this block controls the comparison:

Source: [agent_edit_dpo/evaluate.py](agent_edit_dpo/evaluate.py#L32), lines 32–40. Exact excerpt:

```python
        mode = self.network.disable_adapter() if self.condition == "base" else nullcontext()
        with mode, torch.inference_mode():
            output = self.network.generate(**inputs, max_new_tokens=384, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        completion = output[0, prompt_tokens:]
        raw = self.tokenizer.decode(completion, skip_special_tokens=True)
        return {"message": parse_response(raw, f"local-{self.calls}"), "raw_text": raw,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(completion)},
                "model": self.model, "provider": "local-mps" if self.network.device.type == "mps" else "local-cpu"}
```

For `base`, the context manager temporarily disables the adapter during generation.
For `dpo`, `nullcontext()` leaves it enabled. `do_sample=False` selects greedy decoding.
The raw text and parsed message are both retained in the trace.

The parser accepts exactly one complete JSON object with a supported tool name
and an `arguments` dictionary:

Source: [agent_edit_dpo/protocol.py](agent_edit_dpo/protocol.py#L36), lines 36–47. Exact excerpt:

```python
def parse_response(raw, call_id):
    # Accept only one complete JSON object; malformed text consumes an agent action.
    try:
        value = json.loads(raw.strip())
        if (not isinstance(value, dict) or set(value) != {"tool", "arguments"}
                or value["tool"] not in {"view", "edit", "test", "submit"}
                or not isinstance(value["arguments"], dict)):
            raise ValueError("invalid tool object")
        return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
                "function": {"name": value["tool"], "arguments": json.dumps(value["arguments"])}}]}
    except (ValueError, TypeError):
        return {"role": "assistant", "content": raw}
```

This validates the response's outer structure. The workspace subsequently validates
the actual arguments: a JSON object can parse successfully yet specify an invalid
line range. If parsing fails, the parser returns ordinary assistant text with no
tool call; the reused loop consumes an action and returns corrective feedback.

### Inside the pinned loop: execute, observe, stop, grade

The dependency executes the parsed tool and appends its observation to the conversation:

Source: [pinned aci_patch_agent/agent.py](https://github.com/altaal/aci-patch-agent/blob/42a2683b643b3b9af2f034529032444accf0c849/aci_patch_agent/agent.py#L64), lines 64–78. Exact excerpt:

```python
            try:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                observation = workspace.execute(name, args)
            except (KeyError, TypeError, ValueError):
                observation = {"error": "Malformed tool call. Use a JSON object matching the tool schema."}
            events.append({"action": actions, "response": len(responses), "tool": name,
                           "arguments": args, "observation": observation})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(observation)})
            if name == "submit" and observation.get("submitted"):
                submitted = True
                break
        if submitted:
            break
    evaluation = sandbox.check(task, workspace.source, final=True)
```

This is the pinned implementation used by the saved DPO results. Final evaluation
runs even after the action limit. The returned record separately stores `submitted`,
the evaluation result, and `passed = submitted and evaluation["passed"]`.

`test` runs the two example cases. Final grading runs those examples plus additional
cases. The selection in the reused sandbox is:

Source: [pinned aci_patch_agent/sandbox.py](https://github.com/altaal/aci-patch-agent/blob/42a2683b643b3b9af2f034529032444accf0c849/aci_patch_agent/sandbox.py#L30), lines 30–33. Exact excerpt:

```python
    def check(self, task, source, *, final=False):
        cases = task.examples + task.evaluation if final else task.examples
        # Only inputs go into the container. Expected values stay in the host evaluator.
        inputs = repr([c.args for c in cases])
```

The sandbox writes the current source to a temporary `solution.py` and executes it
in Docker. The host compares returned values/types or exception classes against
expected answers. Those expectations are not supplied to the model or embedded
in its editable source. `run_task` calls the sandbox directly for final grading;
there is no recovery fault wrapper here.

## Exactly how clamp--base--1.json and clamp--dpo--1.json are made

`clamp` is one of the ten evaluation tasks, separate from the training-task IDs.
The shared matrix creates identities using:

Source: [pinned aci_patch_agent/experiment.py](https://github.com/altaal/aci-patch-agent/blob/42a2683b643b3b9af2f034529032444accf0c849/aci_patch_agent/experiment.py#L20), lines 20–22. Exact excerpt:

```python
    attempts = [{"id": f"{task.id}--{condition}--{repeat}", "task": task.id,
                 "condition": condition, "repeat": repeat}
                for repeat in range(1, repeats + 1) for task in tasks for condition in conditions]
```

With task `clamp`, conditions `base` and `dpo`, and repetitions 1–3, this schedules
six independent clamp attempts. For `clamp--dpo--1`, the callback receives the clamp
task, condition `dpo`, and repeat 1. The condition reaches `LocalClient`; the repeat
number labels the attempt and does not select a different seed or training step.

Both recorded repetition-1 attempts started from:

```python
def clamp(value, low, high):
    return min(value, high)
```

Both generated this first response, copied from their saved `raw_text`:

```json
{
  "tool": "edit",
  "arguments": {
    "start": 0,
    "end": 5,
    "replacement": "min(value, high)"
  }
}
```

This parses into an `edit` call, but line 0 is invalid and the source only has two
lines. The workspace returns:

```json
{"error": "Invalid inclusive line range; source has 2 lines. Source unchanged."}
```

These are the actual six events in each of the saved repetition-1 traces:

| Action | Base | DPO |
| --- | --- | --- |
| 1 | Invalid edit range 0–5; source unchanged | Same |
| 2 | No usable tool call | Same |
| 3 | No usable tool call | Same |
| 4 | Repeats invalid edit range 0–5 | Same |
| 5 | No usable tool call | Same |
| 6 | No usable tool call; budget exhausted | Same |

The final source remained the original function. Of the six final checks, three
failed: clamping below the lower bound, clamping a negative range, and raising
`ValueError` for invalid bounds. Both records contain `submitted: false`,
`passed: false`, and `status: "action_limit"`.

Records: [clamp--base--1.json](results/week6-before-after/clamp--base--1.json) and
[clamp--dpo--1.json](results/week6-before-after/clamp--dpo--1.json).

The matrix receives the returned dictionary and adds its task/mode/repetition
identity before serializing it. This is the exact write path:

Source: [pinned aci_patch_agent/experiment.py](https://github.com/altaal/aci-patch-agent/blob/42a2683b643b3b9af2f034529032444accf0c849/aci_patch_agent/experiment.py#L27), lines 27–36. Exact excerpt:

```python
    lookup = {task.id: task for task in tasks}
    def run(attempt):
        try:
            result = runner(lookup[attempt["task"]], attempt["condition"], attempt["repeat"])
        except Exception as error:
            # Preserve the attempt, but don't publish arbitrary exception strings containing paths/keys.
            result = {"passed": False, "status": "runner_error", "error_type": type(error).__name__}
        result.update(attempt)
        (output / f"{attempt['id']}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
```

For the command above, `output` is `runs/week6-published-adapter`, so the write is
`runs/week6-published-adapter/clamp--dpo--1.json`. The committed record under
`results/week6-before-after/` is evidence from the published run, not a file the
new command overwrites. A rerun may produce different model outputs.

After the matrix completes, `summarize_matrix` reads each declared attempt,
treats a missing record as failure, and writes CSV and summary files. Reporting
uses the saved score; it does not invoke the model or rerun the Docker grader.

## Tests, verification, and the actual conclusion

[Unit tests](tests/test_protocol.py) check the text protocol/parser and authored
candidate construction. They do not demonstrate live repair ability. The data
generator runs real behavioral checks on the two authored alternatives, while
the live evaluator runs real checks on model-produced source. The saved-evidence
verifier checks hashes and consistency across recorded files. These are different
checks and answer different questions.

The walkthrough was grounded in local source and saved records. The saved-evidence
verifier passes for the published data, 40-step training run, adapter, DPO source,
and all 60 scored attempts. Documentation work does not rerun training or infer
fresh model performance.

The current result supports this narrow statement: **this dataset, adapter update,
small model, protocol, and budget did not produce completed repairs on these ten
tasks.** It does not establish that DPO generally fails, nor that a lower training
loss improves an agent. The data consists of single edits from five task families,
with easy constant-return negatives; much of the required multi-action behavior
is absent from the preferences. [Full failure analysis](FAILURES.md).

When describing this exercise, keep Week 5's engineering success (verified data
and changed weights) separate from Week 6's capability result (0/30 in both modes).
Any future extension should explain its new question and its exact difference
from this setup before adding implementation or reporting a score.
