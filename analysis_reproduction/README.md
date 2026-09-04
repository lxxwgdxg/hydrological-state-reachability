# Analysis-level reproduction

This directory reproduces the analysis-level evidence for the fixed-weight state-reachability audit reported in **Unreachable water states support streamflow skill in a mass-conserving hydrological model**.

The central result is a controlled distinction between accounting conservation and an executable storage-release trajectory: the audited learner used stepwise unreachable storage trajectories with a measurable streamflow consequence; that consequence survived four prespecified reachable operators, whereas retraining the unchanged architecture with reachable updates removed the dependence and preserved transfer to basins omitted from fitting.

This directory contains self-authored analysis scripts and compact derived tables needed to reproduce the retained statistics and figures. It does not redistribute raw CAMELS-US data, upstream DPL-H source code, or model checkpoints. A separate claim-specific executable replay is provided in `../model_replay/`.

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
5. recompute reachable-trained transfer across 506 basins omitted from fitting, including basin-block uncertainty and leave-one-region influence;
6. rebuild the core and window/initialization figures;
7. compare retained numerical results with the frozen reference file; and
8. scan the package for absolute paths and credential-like literals.

Outputs are written under `outputs/`.

The frozen environment file records the exact versions used for the successful isolated check. Equivalent newer versions may work, but they are outside the verified environment.

## Scope

The uncertainty intervals are conditional on the frozen CAMELS-US samples and three model seeds. They are not global population intervals, independent-dataset validation, cross-architecture validation, or evidence that reachable latent states uniquely correspond to observed hydrological states.

## Model-level reproduction

This directory reproduces the statistical analysis from compact derived tables. The sibling `../model_replay/` directory provides a one-basin, one-checkpoint replay of the executable storage-transition claim over a complete held-out test period. See `MODEL_REPLAY_BOUNDARY.md` for the distinction between these two layers.

## License

Self-authored analysis code is licensed under the MIT License. Self-authored documentation, figures, and compact derived tables are licensed under CC BY 4.0, as specified in the repository-root `LICENSE.md`. Third-party material retains its original licenses and is documented in the repository-root `THIRD_PARTY_NOTICES.md`.
