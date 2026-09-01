# CEE reachability claim-specific model replay v1

This compact package replays the complete 2000-10-01 to 2010-09-30 test period for CAMELS-US basin `03364500` from the frozen seed-11 epoch-30 DPL-H4 checkpoint. It runs the archived model forward path and the same-checkpoint reachability intervention, then compares every frozen basin-level performance, diagnostic and effect field against the accepted reference audit.

This is a **claim-specific model replay**, not a complete retraining package and not evidence beyond the frozen manuscript scope. It deliberately contains one basin and one checkpoint because its purpose is to verify the executable storage-transition claim from model inputs, not to repeat the paper's statistical aggregation.

## Run

From this directory:

```powershell
python run_all.py --device cpu
```

CUDA can be used when available:

```powershell
python run_all.py --device cuda:0
```

The run writes `outputs/model_replay.json`, `outputs/model_replay.csv`, and `outputs/model_replay_verification.json`. A successful verification has `overall_pass: true`, zero frozen-reference mismatches, full 3652-day coverage, and official-forward parity at or below `1e-6 mm day-1`.

## Boundaries

- Upstream DPL-H source is attributed to He et al. and the Zenodo v9 record `https://doi.org/10.5281/zenodo.14515143`. The record API identifies CC BY 4.0.
- The CAMELS-US fixture is attributed to Newman et al. (2015) and Addor et al. (2017); the official UCAR record identifies CC BY 3.0.
- The seed-11 checkpoint and self-authored replay/audit material remain an internal release candidate until authorship and their public license are fixed.
- No credentials, optimizer state or unrelated basin data are included.

See `THIRD_PARTY_NOTICES.md`, `MODEL_REPLAY_BOUNDARY.md`, and `SELF_AUTHORED_LICENSE.md` before any public upload.
