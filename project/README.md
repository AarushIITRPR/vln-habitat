# VLN Project Workspace

This is the clean entry point for the Vision-and-Language Navigation project. The folder is organized by task so a reader can move from setup, to model training, to evaluation, to final submission without digging through the full Habitat repository.

## Start Here

Read `00_README_START_HERE/README.md` first. It explains the whole layout and tells you which folder to open for each purpose.

## Folder Map

```text
project/
├── 00_README_START_HERE/
│   └── Main navigation guide for the workspace
├── 01_Task2_Habitat_Setup/
│   └── Habitat/VLN-CE setup, Intel-safe launchers, simulator output links
├── 02_Task3_CMA_Model/
│   └── CMA model training, checkpoints, logs, videos, report/slides links
├── 03_Task4_Generalization_Ablation/
│   └── Unseen-environment evaluation, paraphrase tests, reduced-data study, ablations
├── 04_Task5_Controlled_Extension/
│   └── Stop-aware controlled extension and baseline comparison
├── 05_Final_Submission/
│   └── Files meant for upload or presentation use
├── 90_Runtime_Dependencies/
│   └── Links to the live VLN-CE, Habitat, dataset, checkpoint, and scene folders
└── 99_Archive/
    └── Old non-submission artifacts kept out of the active workspace
```

## Important Rule

The root-level runtime folders are intentionally still at the repository root:

```text
VLN-CE/
utility_dependencies/
habitat-lab/
habitat-baselines/
data/
```

`utility_dependencies/` contains the actual `data`, `habitat-lab`, and `habitat-baselines` folders. The root-level `data`, `habitat-lab`, and `habitat-baselines` entries are compatibility symlinks so existing training and evaluation scripts keep working.

## Quick Uses

- For final files: open `05_Final_Submission/`.
- For the report and slides: open `02_Task3_CMA_Model/report_and_slides/` or `05_Final_Submission/`.
- For Task 4 graphs/tables: open `03_Task4_Generalization_Ablation/graphs/` and `03_Task4_Generalization_Ablation/tables/`.
- For Task 5 graphs/tables: open `04_Task5_Controlled_Extension/graphs/` and `04_Task5_Controlled_Extension/tables/`.
- For live training/evaluation code: open `90_Runtime_Dependencies/VLN-CE_repository/scripts/`.

## Machine Notes

This workspace was built for Ubuntu Linux with X11 and Intel/Mesa OpenGL. The working Habitat path is CPU/Intel-safe. Do not assume CUDA, an NVIDIA GPU, or `gpu_device_id=0` on this machine.
