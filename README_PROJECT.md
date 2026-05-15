# Current Project Layout

The workspace has been cleaned and organized around:

```text
project/
```

Open this first:

```text
project/00_README_START_HERE/README.md
```

## Main Folders

- `project/01_Task2_Habitat_Setup/`: Habitat/VLN-CE setup and Intel-safe simulator launchers.
- `project/02_Task3_CMA_Model/`: CMA training, model checkpoints, logs, videos, report and slides.
- `project/03_Task4_Generalization_Ablation/`: unseen/generalization, paraphrase, reduced-data, and ablation results.
- `project/04_Task5_Controlled_Extension/`: stop-aware extension results and baseline comparison.
- `project/05_Final_Submission/`: final ZIPs and presentation files.
- `project/90_Runtime_Dependencies/`: links to the live repositories and datasets.
- `project/99_Archive/`: old non-submission files.
- `utility_dependencies/`: bundled Habitat utility folders.

## Why Some Folders Remain At Root

These root folders are still required by runnable scripts:

```text
VLN-CE/
utility_dependencies/
habitat-lab/
habitat-baselines/
data/
```

`utility_dependencies/` contains the actual `data`, `habitat-lab`, and `habitat-baselines` folders. The root-level entries with those names are compatibility symlinks.
