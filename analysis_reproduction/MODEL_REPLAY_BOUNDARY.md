# Model replay boundary

The `analysis_reproduction/` directory rebuilds the retained statistics and figures from compact derived tables. It does not independently reconstruct model checkpoints from raw forcings or repeat model training.

The sibling `model_replay/` directory provides a claim-specific executable check at one frozen checkpoint and one complete held-out test basin. It verifies loading and normalization, numerical parity between the explicit reference transition and the archived model forward method, the same-checkpoint reachability intervention, and the retained basin-level numerical fields.

The full three-seed, 24-basin, and 506-basin results are reproduced from frozen derived tables in the analysis-level directory rather than rerun at model level. Neither layer establishes independent-dataset or cross-architecture generality, or that reachable latent states uniquely correspond to observed hydrological states.

Licensing and attribution for self-authored and third-party material are specified in the repository-root `LICENSE.md` and `THIRD_PARTY_NOTICES.md` and in the corresponding files under `model_replay/`.
