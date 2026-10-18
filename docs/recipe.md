# Training inputs

Pretraining uses `configs/pretrain.json`, a shared source-demonstration split and
training-only normalization. Post-training uses the preceding policy checkpoint,
recorded behavior proposals, judge outputs and an accumulated corpus manifest.

A round manifest contains:

```json
{"corpora": [{"dataset": "/data/round1", "credit_tag": "r1"}, {"dataset": "/data/round2", "credit_tag": "r2"}]}
```

Pass it to `facet0.refinement.train --round-manifest`. Re-encode each corpus with
the same parent policy and its normalization. Score and label every corpus before
policy refinement. Inspect support overflow, per-regime positive rates, reward
components, behavior coverage and auxiliary losses.
