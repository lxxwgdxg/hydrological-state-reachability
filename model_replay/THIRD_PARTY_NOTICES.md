# Third-party notices

## DPL-H archived source

- Creator: Leilei He and collaborators
- Record: https://doi.org/10.5281/zenodo.14515143
- Version: v9, published 2024-12-24
- Archived file: `Code_DPL_Hydrology.zip`
- Archived MD5: `fdc2257eb67e2accaeec7792a572f7eb`
- License recorded by the Zenodo API: CC-BY-4.0
- Local compatibility changes: an otherwise absent package marker was added, and `from __future__ import annotations` was added to `evaluation/plots.py`. Neither changes model or data mathematics. The audited DPL-H4 numerical source hash remains `c0a2402f7c28fc4f199e9dc06969103f48778a9fd4e97146135342ae577ea646`, identical to the archived global DPL-H4 source.

## CAMELS-US

- Newman et al. (2015), DOI: https://doi.org/10.5194/hess-19-209-2015
- Addor et al. (2017), DOI: https://doi.org/10.5194/hess-21-5293-2017
- Official data references: https://doi.org/10.5065/D6MW2F4D and https://doi.org/10.5065/D6G73C3Q
- License identified by the official UCAR record: CC-BY-3.0
- Included subset: one basin's Daymet forcing and USGS discharge plus the seven attribute tables needed by the archived loader.
