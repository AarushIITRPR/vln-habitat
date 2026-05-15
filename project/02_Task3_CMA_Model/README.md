# Task 3: CMA VLN Model

This folder contains the main model work for Task 3: training and evaluating a learned Vision-and-Language Navigation policy.

## What Was Built

The final Task 3 model is based on a Cross-Modal Attention navigation policy. At a high level:

- visual observations are encoded with a convolutional visual encoder;
- language instructions are encoded into contextual token features;
- cross-modal attention fuses instruction and visual state;
- a recurrent policy state tracks navigation history;
- the policy predicts discrete navigation actions using supervised imitation learning.

The model is evaluated through true rollouts from the learned policy. Official evaluation should not use oracle takeover or assisted rollout logic.

## Folder Contents

```text
code_and_scripts/
├── cma_training_script.py
└── navigation_video_script.py

results/
├── model_checkpoints
└── training_logs

report_and_slides/
├── final_report_overleaf.zip
├── final_report_source.tex
├── report_source_files
└── task3_task4_methodology_slides.pptx
```

`code_and_scripts/cma_training_script.py` points to the main CMA training script.

`code_and_scripts/navigation_video_script.py` points to the script used to generate navigation videos and path overlays.

`results/model_checkpoints` links to saved checkpoints.

`results/training_logs` links to training/evaluation logs.

`report_and_slides/` links to the final LaTeX report and methodology slides.

`report_and_slides/report_source_files/` contains the editable source files used to build the report visuals, the Overleaf project, and the methodology slides.

## Metrics Used

Task 3 reports navigation performance using:

- `SR`: Success Rate, the percentage of episodes that stop within the success threshold of the goal.
- `SPL`: Success weighted by Path Length, which rewards successful and efficient navigation.
- `Distance to Goal`: final geodesic or navigation distance remaining at episode end.
- `Action Accuracy`: supervised teacher-forced action prediction accuracy during training/validation.

## Notes For Re-running

Use the Intel-safe launchers in Task 2 or the scripts linked from this folder. Keep Habitat overrides compatible with this machine:

```text
gpu_device_id = -1
concur_render = False
```

Videos are written into the VLN-CE data video folder, linked from Task 2.
