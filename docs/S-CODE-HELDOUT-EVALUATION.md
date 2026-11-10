# S code model independent held-out evaluation

The Slickk source-code training pipeline is in `continuum/repo_training.py`. Its internal training loss does not measure external generalization.

## Windows reproduction

```powershell
cd C:\S
py -3 -m continuum.repo_training --source C:\Slickk-source --output models\s-slickk-source.pt --steps 100
py -3 -m continuum.code_eval --source C:\Slickk-source --checkpoint models\s-slickk-source.pt
py -3 -m unittest discover -s tests -q
```

The evaluator uses the same deterministic whole-file split as training, verifies identical content hashes do not cross splits, and scores a held-out prefix of up to 4096 predicted bytes. It compares causal model negative log-likelihood against a Laplace-smoothed unigram trained only on the training partition. Results include hashes and explicit limitations. This is not a benchmark of code correctness, invention, energy efficiency, or a claim that the model is ready for production. Near-duplicate content across different files remains a potential leakage source.

A checkpoint must actually exist; the evaluator fails rather than fabricating success. CI can run the evaluator's unit tests using a synthetic checkpoint without downloading the Slickk source repository.
