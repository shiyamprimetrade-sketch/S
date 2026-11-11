# S-Brain Cycle 7 — optimizer-connected curriculum, data versioning and blind evaluation

## Scope and verification boundary

Cycle 7 preserves the existing branch, PR #14, Ollama bridge and previous
S-Brain training/inference code. The training workflow is now actually connected
to the curriculum sampler from Cycle 6. The new runner executes neural
forward passes, masked next-byte cross-entropy, backward gradients and AdamW
weight updates in both curriculum and uniform control arms.

No real customer dataset, teacher-generated corpus or model deployment
was performed. CI runs small synthetic fixtures only. Passing tests proves
the software works on those fixtures; it is not evidence that S matches Gemini.

## Modules

- `curriculum_training.py`: train a real independent SBrain from curriculum or
  uniform windows, tracking gradients, weight updates, source IDs, baseline,
  best/final validation checkpoint and all step metrics.
- `curriculum_experiments.py`: paired seed-matched training with both
  strategies using exactly the same model initialization and data.
  Checkpoint selection is *validation only*.
- `dataset_versions.py`: content-addressed dataset metadata with immutable
  release registration; verifies train, validation and blind test fingerprints.
- `blind_benchmarks.py`: independent reviewer-authored frozen benchmark seal,
  source digest checks, held-out token scoring and acceptance gates.
- `experiment_evidence.py`: evidence packets that cross-check experiment
  logs, dataset versions, checkpoint hashes and optional test reports.
- `tests/test_cycle7.py`: actual optimizer training tests, dataset tampering,
  sample leakage, blind seal, matched control and checkpoint integrity.

## Training a real curriculum arm

Requirements: Python with working CPU/CUDA PyTorch and permitted, reviewed
train and validation JSONL files in the Cycle 5 supervised byte format.
From the root of the S repository:

```powershell
python -m s_brain.curriculum_training --train private_train.jsonl --validation private_validation.jsonl --output private_experiments/curriculum --strategy curriculum --steps 100 --context 64 --dim 128 --layers 2 --seed 42
```

For the uniform control:

```powershell
python -m s_brain.curriculum_training --train private_train.jsonl --validation private_validation.jsonl --output private_experiments/uniform --strategy uniform --steps 100 --context 64 --dim 128 --layers 2 --seed 42
```

Each arm writes `baseline/`, `best/` if loss improves, `final/`,
`metrics.jsonl`, `selections.jsonl` and `report.json`. Existing output
directories with files are refused to prevent accidental overwrites.

The two strategies use equivalent weight initialization, optimizer and
validation seeds. The uniform control samples individual supervised windows
uniformly, while the curriculum samples difficulty-weighted records and then
their windows. This difference is intentional and should be disclosed.

## Paired multi-seed experiment

```powershell
python -m s_brain.curriculum_experiments --train private_train.jsonl --validation private_validation.jsonl --output private_experiments/paired --seeds 11 29 47 --steps 100 --context 64 --dim 128 --layers 2
```

Selection is based on the average **best validation cross-entropy** by arm
across the matched seeds. The runner refuses an experiment in which the
baseline losses or dataset fingerprints differ across arms.

The third, independent test split must never be used to decide the better
training strategy. Repeatedly tuning to blind test scores destroys blindness.

## Immutable dataset release

```powershell
python -m s_brain.dataset_versions --train private_train.jsonl --validation private_validation.jsonl --test private_test.jsonl --registry private_versions --name s-learning --version v1 --author human --license user-owned --review manual-reviewed
```

This stores a metadata manifest and source-file hashes, **not** copies of
the source examples. Use a new version ID when the data changes. A registered
version refuses a repeated write and fails verification if the source bytes
change. Do not commit these manifests if they expose sensitive local paths.

## Sealed blind benchmark

Seal a separately authored and reviewed test set before training selection:

```powershell
python -m s_brain.blind_benchmarks seal --test private_test.jsonl --output private_blind_seal.json --name s-blind-v1 --version 1 --author human --reviewer independent-human --license user-owned
```

Evaluate a frozen checkpoint only *after* validation-based selection:

```powershell
python -m s_brain.blind_benchmarks evaluate --test private_test.jsonl --seal private_blind_seal.json --checkpoint private_experiments/paired/seed-000011/curriculum/best --output private_blind_result.json
```

The benchmark measures answer-only teacher-forced byte accuracy and log loss.
It does not measure open-ended correctness, factuality or multistep reasoning.

## Evidence packet

After the experiment and dataset release have been verified:

```powershell
python -m s_brain.experiment_evidence --experiment private_experiments/paired/comparison.json --registry private_versions --name s-learning --version v1 --output private_evidence_packet.json
```

The packet binds training reports to the exact source fingerprints and
checkpoint weights. A packet with missing or mismatched evidence does not
constitute a verified checkpoint improvement.

## Engineering constraints and next milestone

- Training remains independent of Ollama; Ollama is a temporary teacher for
  *reviewed textual examples* only, not a required inference dependency.
- A dataset metadata license field is not itself evidence of ownership.
- The difficulty formula is a heuristic and needs real-world validation.
- The current curriculum sampling is record-weighted rather than
  token-balanced. Results need multi-seed comparisons.
- The blind benchmark is a small technical regression protocol, not an
  independently standardized frontier benchmark.
- The Windows `torch.dll` Code Integrity block is unchanged. Do not weaken
  Windows security to run experiments.

Cycle 8 should focus on non-synthetic benchmark sets, dataset governance,
robustness under distribution shift, genuine memory retrieval, and controlled
S-Tools after the independent model demonstrates useful improvements.
