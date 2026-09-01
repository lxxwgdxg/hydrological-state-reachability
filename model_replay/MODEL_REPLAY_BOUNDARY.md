# Model replay boundary

This package verifies the claim-specific executable path at one frozen checkpoint and one complete test basin.

It does verify:

1. reconstruction of the archived DPL-H4 model from the frozen checkpoint;
2. loading and normalization of the included CAMELS-US basin with the frozen training scaler;
3. exact parity between the manually audited original transition and the official model forward method;
4. exact reproduction of all accepted basin-level original, dormant-ceiling, and ordered-feasible results;
5. the same-checkpoint streamflow consequence and storage-feasibility diagnostics.

It does not verify:

1. model retraining from random initialization;
2. the three-seed or 506-basin statistical aggregation, which is covered by the separate analysis-level package;
3. independent-dataset or cross-architecture generality;
4. the uniqueness or observational truth of the corrected latent state;
5. permission to publish self-authored artifacts before authorship and license are fixed.
