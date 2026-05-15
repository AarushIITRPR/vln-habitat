# Task 4/5 VLN Results

All official rollout metrics use the learned policy only with a 3 m success threshold.

## Key Findings

- Seen validation performance: SR 0.188, SPL 0.188, final distance 8.64 m.
- Unseen validation performance: SR 0.000, SPL 0.000, final distance 9.46 m.
- Paraphrasing reduced seen-split SR from 0.188 to 0.062; unseen SR remained 0.000.
- Increasing clean training data from 32 to 64 trajectories improved SR from 0.000 to 0.250.
- The CMA fusion ablation achieved a lower final distance than gated fusion on the matched 16-episode study (5.92 m vs 12.11 m).
- The Task 5 stop-aware extension improved clean 64-episode SR from 0.125 to 0.250 and SPL from 0.086 to 0.250.

## Interpretation

The results show a clear gap between seen and unseen environments, which is expected for a small CPU-trained VLN setup. The paraphrase test shows that the text encoder handles some lexical variation but remains sensitive to wording. The reduced-data study confirms that imitation quality and route diversity are major bottlenecks. The fusion ablation suggests that cross-modal attention is more useful than a plain gated visual-text merge, especially when judged by final distance even before SR becomes nonzero. The controlled extension mainly helps the model learn when to terminate, which directly improves SR and SPL.
