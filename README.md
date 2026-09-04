# Reachability diagnosis in a mass-conserving hydrological learner

This public repository contains the reproducibility materials associated with the manuscript **Unreachable water states support streamflow skill in a mass-conserving hydrological model**. The repository is maintained at [github.com/lxxwgdxg/hydrological-state-reachability](https://github.com/lxxwgdxg/hydrological-state-reachability), and its releases are archived under the concept DOI [10.5281/zenodo.22221472](https://doi.org/10.5281/zenodo.22221472).

## What can be reproduced

- `analysis_reproduction/` recomputes the unmodified-implementation control, frozen 24-basin and 506-basin statistics, the four-operator lower-envelope result, external transfer of reachable-trained checkpoints, basin-block uncertainty, and spatial influence. It also rebuilds the core and window/initialization figures and verifies all retained values against frozen references.
- `model_replay/` loads one frozen DPL-H4 checkpoint and replays the complete 3,652-day test period for basin `03364500`, including exact reference-forward parity and the same-checkpoint reachability intervention.

Run the complete analysis-level reproduction:

```powershell
python run_all.py
```

Include the model replay when its environment and dependencies are available:

```powershell
python run_all.py --with-model-replay --device cuda:0
```

CPU replay is also supported by replacing `cuda:0` with `cpu`.

## Scientific boundary

This repository supports a claim-specific audit of one mass-conserving recurrent rainfall-runoff architecture across frozen CAMELS-US panels. The fixed-weight intervention tests whether streamflow skill depends on releases that overdraw a named water store; reachable retraining of the unchanged architecture tests whether that dependence is avoidable. The external evaluation frame comprises diverse small-to-medium CAMELS-US basins omitted from fitting and is not presented as global, independent-dataset, or cross-architecture validation.

## Release and licensing boundary

Self-authored source code is licensed under the MIT License. Self-authored documentation, figures, compact derived tables, and trained checkpoint material are licensed under CC BY 4.0. Third-party code and data retain their original licenses and attribution requirements. See `LICENSE.md`, `THIRD_PARTY_NOTICES.md`, and `REPOSITORY_SCOPE.md` for the exact boundaries.
