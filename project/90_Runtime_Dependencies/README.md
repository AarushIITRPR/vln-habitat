# Runtime Dependencies

This folder contains links to the live code, data, checkpoints, logs, and simulator assets. It keeps the project folders tidy while preserving the paths that the scripts expect.

## Contents

```text
VLN-CE_repository
Habitat_Lab
Habitat_Baselines
Root_Data_Links
VLNCE_Data_Checkpoints_Scenes
```

## What Each Link Points To

`VLN-CE_repository` points to the cloned VLN-CE project. This is where the main training, evaluation, graph generation, and video scripts live.

`Habitat_Lab` points to the local Habitat-Lab source tree.

`Habitat_Baselines` points to the Habitat-Baselines package used by the project.

`Root_Data_Links` points to the bundled root data folder.

`VLNCE_Data_Checkpoints_Scenes` points to the VLN-CE data folder, including checkpoints, logs, videos, and dataset-related files.

## Why These Are Links

The actual utility folders are bundled under:

```text
../../utility_dependencies/
```

Habitat and VLN-CE scripts often assume specific relative paths. The workspace root keeps compatibility symlinks, while this folder points directly to the bundled dependencies.

## Machine Compatibility

This machine uses Intel/Mesa OpenGL. Keep simulator launches Intel-safe:

```text
gpu_device_id = -1
concur_render = False
```

Do not assume CUDA, NVIDIA drivers, or CUDA device 0.
