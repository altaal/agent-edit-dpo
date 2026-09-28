"""Build 50 synthetic preferences, each verified using isolated behavioral tests."""

import argparse
import hashlib
import json
from pathlib import Path

from aci_patch_agent.eval_tasks import EVAL_TASKS
from aci_patch_agent.experiment import fingerprint
from aci_patch_agent.sandbox import DockerSandbox
from aci_patch_agent.tasks import TASKS

from .goldens import SOLUTIONS
from .protocol import edit_response, prompt_for, text_messages


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sandbox = DockerSandbox()
    image = sandbox.preflight()
    train_ids, eval_ids = {t.id for t in TASKS}, {t.id for t in EVAL_TASKS}
    assert train_ids.isdisjoint(eval_ids)
    rows, proofs = [], []
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
    (output / "manifest.json").write_text(json.dumps({"pairs": len(rows), "train_pairs": 40,
        "diagnostic_pairs": 10, "independent_training_tasks": 5,
        "training_tasks": sorted(train_ids), "evaluation_tasks": sorted(eval_ids),
        "training_task_sha256": fingerprint(TASKS), "evaluation_task_sha256": fingerprint(EVAL_TASKS),
        "preferences_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "image": image,
        "source": "Authored correct implementations versus constant-return corruptions; no human or model labels.",
        "diagnostic_limit": "Diagnostic pairs share prompts and chosen answers with training. They are not a generalization estimate."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
