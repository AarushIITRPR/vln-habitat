# Workspace Index

This index explains where the project files live and what each area is for.

## Root Workspace

```text
<workspace-root>/
├── project/
├── VLN-CE/
├── utility_dependencies/
├── habitat-lab/
├── habitat-baselines/
└── data/
```

`project/` is the organized human-facing workspace.

`utility_dependencies/` contains the actual utility folders: `data`, `habitat-lab`, and `habitat-baselines`.

The root-level `data`, `habitat-lab`, and `habitat-baselines` entries are compatibility symlinks to `utility_dependencies/`. They stay visible at the root because the training and evaluation scripts depend on those paths.

## Task Folders

| Folder | Purpose |
|---|---|
| `01_Task2_Habitat_Setup/` | Habitat/VLN-CE setup, Intel-safe launchers, simulator output links |
| `02_Task3_CMA_Model/` | CMA model training/evaluation code, checkpoints, logs, videos, report/slides |
| `03_Task4_Generalization_Ablation/` | Generalization, paraphrase, reduced-data, and ablation study outputs |
| `04_Task5_Controlled_Extension/` | Stop-aware extension, baseline comparison, graphs and tables |
| `05_Final_Submission/` | Final files for Overleaf, presentation, and result bundles |
| `90_Runtime_Dependencies/` | Links to live code/data/runtime folders |
| `99_Archive/` | Older non-submission artifacts |

## Main Code Locations

| Need | Location |
|---|---|
| CMA training script | `02_Task3_CMA_Model/code_and_scripts/cma_training_script.py` |
| Navigation video/path overlay script | `02_Task3_CMA_Model/code_and_scripts/navigation_video_script.py` |
| Task 4 rollout evaluation | `03_Task4_Generalization_Ablation/code_and_scripts/rollout_eval_script.py` |
| Task 4 graph/table generation | `03_Task4_Generalization_Ablation/code_and_scripts/task4_graph_generator.py` |
| Task 5 result generation | `04_Task5_Controlled_Extension/code_and_scripts/task5_regularization_graph_generator.py` |
| Full VLN-CE scripts folder | `90_Runtime_Dependencies/VLN-CE_repository/scripts/` |

## Main Result Locations

| Need | Location |
|---|---|
| Final report ZIP | `05_Final_Submission/vln_acm_overleaf.zip` |
| Final methodology slides | `05_Final_Submission/vln_task3_task4_methodology_slides.pptx` |
| Task 4 graphs | `03_Task4_Generalization_Ablation/graphs/final_graphs/` |
| Task 4 tables | `03_Task4_Generalization_Ablation/tables/final_tables/` |
| Task 5 graphs | `04_Task5_Controlled_Extension/graphs/final_graphs/` |
| Task 5 tables | `04_Task5_Controlled_Extension/tables/final_tables/` |
| Simulator videos | `01_Task2_Habitat_Setup/outputs/simulator_videos/` |

## Report And Presentation Source

Editable report and slide source files are in:

```text
02_Task3_CMA_Model/report_and_slides/report_source_files/
```

The ready-to-upload/report files are also copied into:

```text
05_Final_Submission/
```

## Runtime Data And Checkpoints

Use these links when checking runtime outputs:

```text
90_Runtime_Dependencies/VLNCE_Data_Checkpoints_Scenes/
../utility_dependencies/
02_Task3_CMA_Model/results/model_checkpoints/
02_Task3_CMA_Model/results/training_logs/
```

## Recommended Navigation

For reading the project: start at `project/README.md`, then follow the numbered task folders.

For submission: use only `05_Final_Submission/`.

For debugging or rerunning: use `90_Runtime_Dependencies/VLN-CE_repository/` and the task-specific script links.

For old material: use `99_Archive/`, but do not mix it into the current submission.
