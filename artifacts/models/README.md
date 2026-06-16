# Trained Model Artifacts

Training scripts write checkpoints here by default.

```text
artifacts/models/
  multisite_current/
    jumprs/
      model.pt
      metadata.json
      training_history.csv
    patchtst/
    itransformer/
    timesnet/
    quantilegru/
    mc_dropout/
    timediff_style/
    nsdiff_style/
    persistence/
  pvdaq_system_34_rollout_4h/
    ...
```

`model.pt` files are ignored by normal Git because they are binary artifacts.
Use Git LFS or a release asset if you need to publish trained weights.

