# Benchmark Implementations

Each benchmark has its own folder-level entry point. The implementation is kept
thin and delegates shared training utilities to `src.models.multistep_baselines`
so that all benchmarks use the same preprocessing, split protocol, and metric
calculation.

```text
src/benchmarks/
  persistence/
  quantile_gru/
  mc_dropout/
  patchtst/
  itransformer/
  timesnet/
  timediff/
  nsdiff/
```

