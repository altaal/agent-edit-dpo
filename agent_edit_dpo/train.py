"""One fixed DPO run with a LoRA adapter. No evaluation-task tuning."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=".model")
    parser.add_argument("--output", default="training/main")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer
    set_seed(7)
    rows = [json.loads(line) for line in Path("data/preferences.jsonl").read_text().splitlines()]
    dataset = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows if r["split"] == "train"])
    source = json.loads(Path("model-source.json").read_text())
    configuration = {"steps": args.steps, "seed": 7, "beta": 0.1, "learning_rate": 1e-5,
        "batch_size": 1, "gradient_accumulation_steps": 1, "max_length": 1024,
        "lora_rank": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"],
        "dtype": "float32", "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "model": source, "preferences_sha256": hashlib.sha256(Path("data/preferences.jsonl").read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "trl", "peft", "datasets", "accelerate"]}}
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.config.use_cache = False
    # Keep the adapter's public provenance independent of this machine's cache path.
    model.name_or_path = source["model_id"]
    model.config._name_or_path = source["model_id"]
    config = DPOConfig(output_dir=str(output), max_steps=args.steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-5, beta=0.1, max_length=1024,
        bf16=False, fp16=False, gradient_checkpointing=True, optim="adamw_torch",
        precompute_ref_log_probs=True, precompute_ref_batch_size=1,
        save_strategy="no", logging_steps=1, report_to=[], seed=7, data_seed=7,
        dataloader_pin_memory=False, disable_tqdm=True)
    started = time.monotonic()
    trainer = DPOTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer,
        peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                              lora_dropout=0.0, task_type="CAUSAL_LM"))
    # Check that no training sequence is truncated, including its assistant answer.
    lengths = []
    for row in rows:
        for choice in ["chosen", "rejected"]:
            lengths.append(len(tokenizer.apply_chat_template(row["prompt"] + row[choice], tokenize=True)))
    if max(lengths) > config.max_length:
        raise ValueError(f"Training sequence exceeds fixed context: {max(lengths)}")
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
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
