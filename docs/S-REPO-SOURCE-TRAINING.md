# S source-code weight-training experiment

Source: `shiyamprimetrade-sketch/Slickk`, branch `slickk-native-runtime`. This is a **separate** project; no source repo modifications.

This pipeline builds a code corpus from an explicitly checked-out local repository. It excludes common build artifacts, data, hidden paths, large files, and credential-looking files. SHA-256 of each file determines its train/evaluation partition; entire files remain in one partition. This is not sufficient to prevent near-duplicate leakage across different files; inspect manually before relying on benchmark claims.

## Windows commands
```powershell
cd C:\
git clone --branch slickk-native-runtime --single-branch https://github.com/shiyamprimetrade-sketch/Slickk.git Slickk-source
cd C:\S
py -3 -m pip install torch
py -3 -m continuum.repo_training --source C:\Slickk-source --prepare-only
py -3 -m continuum.repo_training --source C:\Slickk-source --output models\s-slickk-source.pt --steps 100
```

The output includes a PyTorch checkpoint and manifest. The byte model is intentionally tiny and not a usable programming assistant without further work. The current training function internally measures a validation sample drawn from training files; the external held-out files are **reserved but not evaluated**. No claim of improved reasoning, code generation, patent novelty, or readiness for deployment is justified by the checkpoint alone.

The source repository's own training pipeline (`server/src/training/corpus.js`) plans educator-reviewed curriculum tasks and declares `readyForWeightTraining: false`; those tasks are not pre-trained model weights. Its manifest/evaluation gating (`server/src/training/registry.js`) informs this pipeline's provenance and evaluation design. Do not import curriculum or personal data as code training without consent and applicable rights.
