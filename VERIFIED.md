# Release verification

September 28, 2026, from a new clone of public commit `99aa1d3` and a new
Python 3.12 environment:

- Installed the package and pinned training dependencies successfully.
- All three protocol/data unit tests passed.
- The offline verifier passed: 50 preference records, 40 training steps, disjoint
  training/evaluation tasks, unchanged data/source/adapter hashes, and all 60 scored
  attempts with the correct denominators.
- The saved before/after report regenerated without a Git diff.
- Loaded the committed adapter on the pinned base model using the previously
  downloaded base-weight cache. All 96 loaded tensors exactly matched the saved
  checkpoint, containing 540,672 parameters. This was a fresh environment/load
  check, not another complete training or evaluation run.
- GitHub Actions passed package installation, unit tests, and the offline verifier
  on Linux/Python 3.12. CI does not download model weights or train a model.
- Gitleaks found no secrets in the publication history.

The first local installation probe incorrectly quoted a shell variable with an
extras suffix, so installation failed before testing. Repeating it from the clone
directory with the documented literal `'.[train]'` argument succeeded. Subsequent
verification used stop-on-error execution so a failed installation could not be
mistaken for a completed check.

The original 60 local agent attempts remain the evaluation record. No successful
attempts were substituted, and the negative before/after result is unchanged.
