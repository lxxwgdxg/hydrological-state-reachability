# Claim-specific model replay

This directory replays the complete 2000-10-01 to 2010-09-30 test period for CAMELS-US basin `03364500` from the frozen seed-11 epoch-30 DPL-H4 checkpoint. It loads the archived model forward path, verifies its numerical parity with the explicit reference transition used in the audit, and then applies the same-checkpoint reachability intervention. Every retained basin-level performance, diagnostic, and effect field is compared with the frozen reference audit.

This is a **claim-specific model replay**, not a complete retraining package and not evidence beyond the frozen manuscript scope. It deliberately contains one basin and one checkpoint because its purpose is to verify the executable storage-transition claim from model inputs, not to repeat the paper's three-seed and multi-basin statistical aggregation.

## Run

From this directory:

```powershell
python run_all.py --device cpu
```

CUDA can be used when available:

```powershell
python run_all.py --device cuda:0
```

The run writes `outputs/model_replay.json`, `outputs/model_replay.csv`, and `outputs/model_replay_verification.json`. A successful verification has `overall_pass: true`, zero frozen-reference mismatches, full 3,652-day coverage, and reference-forward parity at or below `1e-6 mm day-1`.

## Boundaries

- The upstream DPL-H source is attributed to He et al. and the Zenodo v9 record [10.5281/zenodo.14515143](https://doi.org/10.5281/zenodo.14515143). The Zenodo record identifies the license as CC BY 4.0.
- The included CAMELS-US fixture is attributed to Newman et al. (2015) and Addor et al. (2017); the official UCAR records identify the source license as CC BY 3.0.
- The seed-11 checkpoint and self-authored documentation are licensed under CC BY 4.0. Self-authored replay and audit code is licensed under the MIT License.
- No credentials, optimizer state, or unrelated basin data are included.

See `THIRD_PARTY_NOTICES.md`, `MODEL_REPLAY_BOUNDARY.md`, `SELF_AUTHORED_LICENSE.md`, and the repository-root `LICENSE.md` for the exact attribution, scope, and licensing boundaries.
