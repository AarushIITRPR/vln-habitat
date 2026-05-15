# Task 5: Controlled Extension

## Extension Chosen

The selected extension is regularization, implemented as stop-aware imitation learning. In VLN, success requires the policy to explicitly choose STOP within the success radius. Since STOP is sparse in demonstration trajectories, the baseline action objective can under-train termination behavior.

## Implementation

The extension uses two small training-objective changes:

- `ACTION_LOSS_WEIGHTS=1.5,1.0,1.0,1.0`
- `STOP_ACTION_REPEAT=4`

This keeps the CMA architecture unchanged and only regularizes the action supervision so the model receives stronger signal around termination.

## Quantitative Comparison

| Model | SR | SPL | Distance to Goal |
|---|---:|---:|---:|
| Baseline CMA | 0.125 | 0.086 | 9.478 m |
| Stop-aware regularized CMA | 0.250 | 0.250 | 7.198 m |

The extension improves SR by 0.125, SPL by 0.164, and reduces final distance by 2.280 m.

## Analysis

The improvement shows that rollout success was limited not only by path following, but also by termination calibration. The baseline often learned plausible movement actions but failed to stop reliably. Stop-aware regularization made the terminal decision more visible during training, which improved both SR and SPL without changing the policy architecture. This is a controlled extension because the model, data size, encoder setup, and CMA fusion mechanism remain fixed while only the loss supervision is changed.
