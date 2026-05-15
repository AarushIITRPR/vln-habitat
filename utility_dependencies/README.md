# Utility Dependencies

This folder bundles the large utility/runtime folders used by the VLN project.

## Contents

```text
utility_dependencies/
├── data/
├── habitat-baselines/
└── habitat-lab/
```

`data/` contains root-level Habitat data links/assets used outside the VLN-CE repository.

`habitat-lab/` contains the Habitat-Lab source tree.

`habitat-baselines/` contains Habitat-Baselines source/configuration code.

## Compatibility Links

The workspace root still has these symlinks:

```text
data -> utility_dependencies/data
habitat-lab -> utility_dependencies/habitat-lab
habitat-baselines -> utility_dependencies/habitat-baselines
```

Those links are intentional. Several scripts refer to paths such as `../habitat-lab` and `../habitat-baselines` from inside `VLN-CE/`, so the root-level symlinks preserve compatibility while keeping the actual folders bundled here.
