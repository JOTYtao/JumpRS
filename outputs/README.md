# Outputs

This directory stores generated metrics, figures, logs, and prediction files.

Prediction CSV archives can be several gigabytes and are ignored by normal Git.
Regenerate them with:

```bash
python -m src.run_multisite_experiments
python -m src.run_pvdaq34_rollout_experiments
python -m src.evaluation.compute_pvdaq34_lead_metrics
python -m src.visualization.paper_figures
```

