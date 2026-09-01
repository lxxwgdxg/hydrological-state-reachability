# Model replay boundary

The audited DPL-H source archive is available from Zenodo DOI `10.5281/zenodo.14515143`, associated with He et al. (2024), DOI `10.1029/2024WR037582`.

At the time this candidate was built, the Zenodo record exposed a Rights/License heading without a concrete license value, and the downloaded top-level code directory contained no LICENSE, COPYING or NOTICE file. The upstream source is therefore not redistributed here.

Full model replay additionally requires:

- obtaining the upstream archive from its original source;
- obtaining CAMELS-US forcing, discharge and attribute data under their source terms;
- applying the documented self-authored reachability changes or patch;
- using the frozen 24-basin and 506-basin lists, seeds and epoch-30 checkpoint identities;
- either obtaining permitted checkpoint files or retraining from the fixed configurations.

Until licensing, path portability and checkpoint-distribution reviews are complete, this directory must not be described as a self-contained model-level reproduction package.
