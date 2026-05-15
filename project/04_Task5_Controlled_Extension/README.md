# Task 5: Controlled Extension

This folder contains the controlled extension work for the project.

## Extension Implemented

Task 5 adds a stop-aware improvement to the baseline navigation policy. The goal is to reduce premature stopping and missed stopping near the goal by making the policy more sensitive to when the agent should issue `STOP`.

This is a controlled extension because it changes one targeted behavior while keeping the rest of the evaluation comparable to the baseline.

## Folder Contents

```text
code_and_scripts/
└── task5_regularization_graph_generator.py

results/
├── task5_full_results
└── task5_results_bundle.zip

graphs/
└── final_graphs

tables/
└── final_tables
```

`code_and_scripts/task5_regularization_graph_generator.py` points to the script that generates the Task 5 comparison graphs and tables.

`results/task5_full_results` links to the complete extension result folder.

`graphs/final_graphs` links to the final Task 5 visualizations.

`tables/final_tables` links to the quantitative comparison and written result discussion.

## What To Compare

Task 5 should be read as a baseline-versus-extension comparison:

- baseline CMA policy;
- CMA policy with stop-aware improvement;
- changes in SR, SPL, final distance to goal, and stopping behavior.

The figures and tables in this folder are intended for the project report and presentation.
