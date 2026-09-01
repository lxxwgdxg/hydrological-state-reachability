# Reachability diagnosis in a mass-conserving hydrological learner

This is an analysis-level reproduction candidate for the retained single claim:

> Algebraic mass balance did not prevent the audited hydrological learner from using stepwise unreachable storage trajectories; the fixed-weight consequence survived four reachable operators, whereas retraining the unchanged architecture removed the dependence and preserved transfer to basins omitted from fitting.

The package contains only self-authored analysis scripts and compact derived tables needed to reproduce the retained statistics and figures. Upstream DPL-H source code, raw CAMELS-US data, model checkpoints and credentials are not redistributed here.

## Run

From this directory:

```powershell
python run_all.py
```

The command will:

1. reproduce the unmodified seed-11 implementation control and its batch-context diagnosis;
2. recompute the 24-basin uncertainty and influence statistics;
3. recompute the 506-basin transport and regional robustness statistics;
4. recompute the four-operator lower-envelope robustness result;
5. recompute reachable-trained transfer across 506 omitted basins, including basin-block uncertainty and leave-one-region influence;
6. rebuild the core and window/initialization figures;
7. compare retained numerical results with the frozen reference file;
8. scan the package for absolute paths and credential-like literals.

Outputs are written under `outputs/`.

The frozen environment file records the exact versions used for the successful
isolated check. Equivalent newer versions may work, but are not part of this
audited candidate.

## Scope

The uncertainty intervals are conditional on the frozen CAMELS-US samples and three model seeds. They are not global population intervals, independent-dataset validation, cross-architecture validation, or evidence that reachable latent states uniquely equal observed hydrological states.

## Model-level reproduction

This package reproduces the analysis from compact derived tables. Full model replay has separate data, source-code and license requirements; see `MODEL_REPLAY_BOUNDARY.md`.

## License status

No public license is granted by this draft package. A license will be selected only after authorship and third-party dependency boundaries are finalized.
