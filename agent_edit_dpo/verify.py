"""Check the committed evidence without downloading or running a model."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data = read("data/manifest.json")
    pairs = [json.loads(line) for line in Path("data/preferences.jsonl").read_text().splitlines()]
    proofs = read("data/verification.json")
    require(len(pairs) == len(proofs) == 50, "Expected 50 preference pairs and proofs")
    require(len({p["id"] for p in pairs}) == 50, "Duplicate preference IDs")
    require({p["id"] for p in pairs} == {p["id"] for p in proofs}, "Pair/proof IDs differ")
    require(sum(p["split"] == "train" for p in pairs) == 40, "Expected 40 training pairs")
    require(sum(p["split"] == "diagnostic" for p in pairs) == 10, "Expected 10 diagnostic pairs")
    require(all(p["chosen"]["passed"] and not p["rejected"]["passed"]
                and p["chosen"]["error"] is None and p["rejected"]["error"] is None
                for p in proofs), "Preference labels lack successful verification records")
    data_hash = hashlib.sha256(Path("data/preferences.jsonl").read_bytes()).hexdigest()
    require(data_hash == data["preferences_sha256"] == read("training/main/config.json")["preferences_sha256"],
            "Preference data hash differs from training input")
    require(set(data["training_tasks"]).isdisjoint(data["evaluation_tasks"]), "Training/evaluation task overlap")
    metrics = read("training/main/metrics.json")
    require(metrics["global_step"] == 40 and metrics["changed_parameter_tensors"] > 0,
            "Final training did not record 40 updates and changed weights")
    output = Path("results/week6-before-after")
    manifest, summary = read(output / "manifest.json"), read(output / "summary.json")
    require(len(manifest["attempts"]) == 60, "Expected 60 declared attempts")
    require(len({a["id"] for a in manifest["attempts"]}) == 60, "Duplicate attempt IDs")
    require(manifest["task_sha256"] == data["evaluation_task_sha256"], "Evaluation task definitions changed")
    weight_hash = hashlib.sha256(Path("training/main/adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    require(weight_hash == manifest["adapter_sha256"] == read("training/main/provenance.json")["adapter_sha256"],
            "Evaluated adapter differs from saved training weights")
    for name, expected in manifest["source_sha256"].items():
        require(hashlib.sha256((Path("agent_edit_dpo") / name).read_bytes()).hexdigest() == expected,
                f"Evaluated source changed: {name}")
    rows = []
    for attempt in manifest["attempts"]:
        result = read(output / (attempt["id"] + ".json"))
        require(all(result[key] == value for key, value in attempt.items()), "Attempt identity mismatch")
        require(result["max_actions"] == manifest["max_actions"] == 6, "Action budgets differ")
        require(result["passed"] == (result["submitted"] and result["evaluation"]["passed"]),
                "Published score disagrees with submission/final checks")
        rows.append(result)
    for condition in ["base", "dpo"]:
        group = [r for r in rows if r["condition"] == condition]
        require(len(group) == summary[condition]["attempts"] == 30, "Incorrect denominator")
        require({r["task"] for r in group} == set(data["evaluation_tasks"]), "Incorrect evaluation task coverage")
        require(all(sum(r["task"] == task for r in group) == 3 for task in data["evaluation_tasks"]),
                "Expected three attempts per task and condition")
        require(sum(r["passed"] for r in group) == summary[condition]["passed"], "Incorrect pass count")
        require(summary[condition]["pass_rate"] == sum(r["passed"] for r in group) / 30, "Incorrect pass rate")
    print("Verified saved evidence: 50 preference records, 40 training steps, matching weights/source, and 60 scored attempts.")
    print("This checks artifact consistency; it does not rerun training, the model, or behavioral tests.")


if __name__ == "__main__":
    main()
