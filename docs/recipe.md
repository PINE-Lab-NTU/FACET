# Refinement rounds (Alg. 2)

Round 1 starts from `pi_align`. Each later round starts from the previous `pi_ref`, adds
the new rollouts to D, re-encodes every corpus in D with the current policy and
retrains the critic on all of them.

Set `OPENPI_SRC`, `DATASET` (this round's rollouts) and `ROUND` first (`scripts/env.sh`).
Keep datasets and checkpoints on a local SSD.

| Step | Script | Output | Check before moving on |
| --- | --- | --- | --- |
| 0. ingest rollouts | `prepare_rollouts.sh` | RL-ready corpus, `facet0/norm_stats.json` | audit report success / failure counts; state stats 67-D, action stats 13-D |
| 1. rewards | `rewards.sh` | `meta/returns_{ROUND}.parquet` | rewards at sub-goal completions; no reward on failed outcome-only rollouts |
| 2. encode | `encode.sh` (GPU) | `meta/features_{ROUND}/` | `meta.json` lists the `pi_ref` params; frame count equals the corpus |
| 3. critic | `train_critic.sh` | critic directory | `td` falls; `value_std` grows; `ranking_pairs > 0`; `contact` loss below its initial value |
| 4. credit | `credit.sh` | `meta/values_{ROUND}.parquet`, `meta/credit_{ROUND}.parquet` + `.json` | `positive_rate_contact` and `positive_rate_free` both at the configured fraction; positives spread over successful episodes rather than piled at their start |
| 5. refinement | `refine_policy.sh` | `pi_ref` checkpoints | log line `Loaded norm stats`; `conditional_loss` and `unconditional_loss` fall; `contact_weight_mean` above 1 |
| 6. deployment | `serve.sh` | websocket policy | clients send `observation/wrench_history`; actions at guidance scale 0 and 1 differ |

Later rounds, with D = {r1, r2}:

```bash
# re-encode r1 with the new policy, encode r2, train on both
DATASET=/data/r1 ROUND=r2 PI_REF=<pi_ref r1>/params FEATURES=/data/r1/meta/features_r2 RETURNS_TAG=r1 bash scripts/encode.sh
DATASET=/data/r2 ROUND=r2 PI_REF=<pi_ref r1>/params bash scripts/encode.sh
DATASET=/data/r2 ROUND=r2 CRITIC=critics/r2 \
  CORPORA="/data/r1:/data/r1/meta/features_r2:r1 /data/r2:/data/r2/meta/features_r2:r2" bash scripts/train_critic.sh
```

Notes:

* Each training sample decodes three 480 x 640 videos. On a spinning disk the data
  loader starves the accelerator; data-loader workers stuck in `D` state are the sign.
* Encoding cost is one guided sampling pass per frame and proposal. `--limit` and
  `NUM_PROPOSALS` trade coverage for time.
* When normalization statistics change between rounds, initialize from the weights
  (`PI_REF`) with a fresh optimizer rather than resuming.
