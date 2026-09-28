"""Compare the base model and trained adapter in the same bounded tool loop."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from aci_patch_agent.agent import run_task
from aci_patch_agent.eval_tasks import EVAL_TASKS
from aci_patch_agent.experiment import execute_matrix, summarize_matrix
from aci_patch_agent.sandbox import DockerSandbox

from .protocol import parse_response, text_messages, PROTOCOL


class LocalClient:
    def __init__(self, model, tokenizer, condition):
        self.network, self.tokenizer, self.condition = model, tokenizer, condition
        self.model = "Qwen/Qwen2.5-0.5B-Instruct:" + condition
        self.calls = 0

    def complete(self, messages):
        import torch
        self.calls += 1
        text = self.tokenizer.apply_chat_template(text_messages(messages), tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.network.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens > 8192:
            from aci_patch_agent.client import ModelError
            raise ModelError("Local context exceeds 8192 tokens")
        mode = self.network.disable_adapter() if self.condition == "base" else nullcontext()
        with mode, torch.inference_mode():
            output = self.network.generate(**inputs, max_new_tokens=384, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        completion = output[0, prompt_tokens:]
        raw = self.tokenizer.decode(completion, skip_special_tokens=True)
        return {"message": parse_response(raw, f"local-{self.calls}"), "raw_text": raw,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(completion)},
                "model": self.model, "provider": "local-mps" if self.network.device.type == "mps" else "local-cpu"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=".model")
    parser.add_argument("--adapter", default="training/main/adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        print(summarize_matrix(args.output))
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(7)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    sandbox = DockerSandbox()
    image = sandbox.preflight()
    metadata = {"experiment": "base-versus-dpo", "model": json.loads(Path("model-source.json").read_text()),
        "agent_revision": "42a2683b643b3b9af2f034529032444accf0c849", "device": device,
        "max_actions": 6, "max_total_tokens": 32000, "max_new_tokens": 384, "do_sample": False,
        "seed": 7, "text_tool_protocol": PROTOCOL, "image": image,
        "adapter_sha256": hashlib.sha256((Path(args.adapter) / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}}
    print(execute_matrix(args.output, EVAL_TASKS, ["base", "dpo"], 3,
        lambda task, condition, repeat: run_task(task, LocalClient(model, tokenizer, condition), sandbox,
                                                max_actions=6), metadata, workers=1))


if __name__ == "__main__":
    main()
