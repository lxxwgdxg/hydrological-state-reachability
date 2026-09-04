# Model replay boundary

This directory verifies the claim-specific executable path at one frozen checkpoint and one complete held-out test basin.

It verifies:

1. reconstruction of the archived DPL-H4 model from the frozen checkpoint;
2. loading and normalization of the included CAMELS-US basin with the frozen training scaler;
3. numerical parity between the explicit reference transition and the archived model forward method;
4. exact reproduction of all retained basin-level original, dormant-ceiling, and ordered-feasible results; and
5. the same-checkpoint streamflow consequence and storage-reachability diagnostics.

It does not verify:

1. model retraining from random initialization;
2. the three-seed or 506-basin statistical aggregation, which is reproduced by the separate analysis-level directory;
3. independent-dataset or cross-architecture generality;
4. the uniqueness or observational truth of the reachable latent state; or
5. process correspondence beyond the tested storage-release constraint.

The included self-authored and third-party materials are governed by `SELF_AUTHORED_LICENSE.md`, `THIRD_PARTY_NOTICES.md`, and the repository-root `LICENSE.md`.
