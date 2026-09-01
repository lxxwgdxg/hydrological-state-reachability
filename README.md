
# Reachability diagnosis in a mass-conserving hydrological learner

This repository is the clean public-release staging package associated with the Article **Unreachable water states support streamflow skill in a mass-conserving hydrological model**. Nothing has been uploaded. Personal names and contact details are intentionally omitted from the repository files. The public repository URL and archival DOI will be inserted when the deposition record is created.

## What can be reproduced

- `analysis_reproduction/` recomputes the unmodified-implementation control, frozen 24-basin and 506-basin statistics, four-operator lower-envelope result, external transfer of reachable-trained checkpoints, block uncertainty and spatial influence; it rebuilds the core and window/initialization figures and verifies all retained values against frozen references.
- `model_replay/` reconstructs one frozen DPL-H4 checkpoint and replays the complete 3,652-day test period for basin 03364500, including exact official-forward parity and the same-checkpoint reachability intervention.

Run the complete analysis-level reproduction:

```powershell
python run_all.py
```

Include the model replay when the model environment and data dependencies are available:

```powershell
python run_all.py --with-model-replay --device cuda:0
```

CPU replay is also supported by replacing `cuda:0` with `cpu`.

## Scientific boundary

The repository reproduces a claim-specific audit of one mass-conserving hydrological architecture across frozen CAMELS-US panels; the source data report the sampled hydroclimatic and catchment-size frame. The external frame contains diverse small-to-medium CAMELS-US basins omitted from fitting and is not presented as global or independent-dataset validation.

## Release boundary

The repository contains the analysis and replay assets described above. Licensing and third-party boundaries are recorded in `THIRD_PARTY_NOTICES.md`, `REPOSITORY_SCOPE.md` and `LICENSE.md`.
