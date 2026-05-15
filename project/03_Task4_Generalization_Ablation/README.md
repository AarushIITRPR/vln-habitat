# Task 4: Generalization and Ablation Study

This folder contains the Task 4 evaluation artifacts.

## What Task 4 Covers

Task 4 studies how the learned VLN policy behaves beyond the simplest training setup:

- evaluation on unseen environments;
- testing with paraphrased instructions;
- performance under reduced training data;
- at least one ablation experiment comparing model variants.

## Folder Contents

```text
code_and_scripts/
├── rollout_eval_script.py
└── task4_graph_generator.py

results/
├── task4_full_results
└── task4_results_bundle.zip

graphs/
└── final_graphs

tables/
└── final_tables
```

`code_and_scripts/rollout_eval_script.py` points to the rollout evaluation script.

`code_and_scripts/task4_graph_generator.py` points to the graph/table generation script.

`results/task4_full_results` links to the complete Task 4/Task 5 result folder.

`graphs/final_graphs` links to the generated presentation/report figures.

`tables/final_tables` links to generated CSV and Markdown result tables.

## Interpreting The Results

The important comparison dimensions are:

- train/seen split performance versus unseen split performance;
- original instructions versus paraphrased instructions;
- full-data training versus reduced-data training;
- baseline model versus ablated model variant.

Use the graphs for presentation slides and the tables for exact values in the report.
