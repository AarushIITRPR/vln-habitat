# Start Here

This folder is the workspace guide. The actual project is organized into numbered task folders so the flow matches the assignment.

For the detailed file-by-file map, open `WORKSPACE_INDEX.md`.

## What To Open First

Use this order if you are trying to understand the project from scratch:

1. `01_Task2_Habitat_Setup/`
2. `02_Task3_CMA_Model/`
3. `03_Task4_Generalization_Ablation/`
4. `04_Task5_Controlled_Extension/`
5. `05_Final_Submission/`

## What Each Task Contains

`01_Task2_Habitat_Setup/` contains the Habitat and VLN-CE setup links, Intel-safe launcher scripts, and simulator video output location.

`02_Task3_CMA_Model/` contains the main VLN model work: CMA training script, evaluation/video script, checkpoints, logs, and report/slides links.

`03_Task4_Generalization_Ablation/` contains the generalization and ablation study results: seen/unseen evaluation, paraphrased instruction tests, reduced-data comparisons, graphs, and tables.

`04_Task5_Controlled_Extension/` contains the controlled extension: the stop-aware improvement, quantitative comparison, graphs, tables, and discussion outputs.

`05_Final_Submission/` contains the presentation/report artifacts that are ready to upload or share.

`90_Runtime_Dependencies/` contains links to the live repositories and datasets. It exists so the task folders stay clean while the underlying Habitat paths remain runnable.

`99_Archive/` contains older artifacts that are not part of the active submission.

## Most Useful Locations

```text
project/05_Final_Submission/
```

Final ZIPs, slides, and result bundles.

```text
project/90_Runtime_Dependencies/VLN-CE_repository/scripts/
```

Training, evaluation, graph generation, and video generation scripts.

```text
project/90_Runtime_Dependencies/VLNCE_Data_Checkpoints_Scenes/
```

VLN-CE data, checkpoints, logs, videos, and scene-related runtime files.

## Do Not Move These Root Folders

The following folders are deliberately outside `project/`:

```text
VLN-CE/
utility_dependencies/
habitat-lab/
habitat-baselines/
data/
```

`utility_dependencies/` contains the actual `data`, `habitat-lab`, and `habitat-baselines` folders. The root-level entries with those names are compatibility symlinks, because scripts expect those paths to remain valid.
