# Setup, training and serving

Joint action–wrench pretraining, value-guided policy refinement, and policy serving.
The local actor and robot runtime are provided by `facet-online-adaptation`.

## Environment

Python 3.11. The JAX policy and PyTorch robot client may use separate environments.

```bash
# From the FACET repository root.
git clone https://github.com/Physical-Intelligence/openpi.git ../openpi
git -C ../openpi checkout 650c5b0283a49c42784fb5055a0507da2c6d347d
pip install --no-deps -e ../openpi -e ../openpi/packages/openpi-client
pip install -e '.[policy,test]' -e './facet-online-adaptation[test]'
export PYTHONPATH="$PWD:$PWD/facet-online-adaptation/src:$PWD/../openpi/src:$PWD/../openpi/packages/openpi-client/src"
```

## Pretraining

Robot episodes use the LeRobot format, `source_demo_id`, and an explicit `tier`
from `base`, `precision`, or `recovery`. VQA JSONL records contain `image`,
`question`, `answer`, and `source_episode_index`.

```bash
python -m facet0.data.split --dataset data/demonstrations --out data/split.json
python -m facet0.data.norm_stats --dataset data/demonstrations --split data/split.json --out assets/normalization
python -m facet0.pretraining.initialize --params /path/to/base/params --norm-stats assets/normalization --out checkpoints/initial
python -m facet0.pretraining.train --config configs/pretrain.json
python -m facet0.pretraining.evaluate --config configs/pretrain.json --checkpoint checkpoints/pretraining/40000
```

`steps` counts optimizer updates: four action–wrench updates alternate with one VQA
update. Each optimizer keeps its own state and warmup counter. `resume` points to a
numeric pretraining checkpoint; its data split, normalization and optimizer configuration
must match. Checkpoints contain `params/`, `assets/norm_stats.json`, `manifest.json`
and `training.pkl`. Load training state only from trusted files.

## Post-training

Rollouts contain the applied `action`, judge columns `subgoal_index`,
`subgoal_complete`, `phase`, and `behavior_joint_proposal[50,13]` in physical
units and the corpus gripper convention. Its first action matches `requested_command`;
`action` stores the command accepted by the execution map. The remaining rows
describe the logged plan. `facet0.data.transition.behavior_fields` validates identities
and pre-execution timestamps. Each transition spans one recorded
control interval. The proposal is recorded before its outcome is observed.

```bash
export OPENPI_SRC=/path/to/openpi/src DATASET=/data/round1 ROUND=r1
RAW=/data/raw_rollouts bash scripts/prepare_rollouts.sh
ALPHA=0.5 BETA=0.2 TIME_BUDGET=450 ENVELOPES=configs/module.envelopes.json bash scripts/rewards.sh
PI_REF=/path/to/pi_align/params bash scripts/encode.sh
CRITIC=critics/r1 bash scripts/train_critic.sh
CRITIC=critics/r1 bash scripts/credit.sh
PI_REF=/path/to/pi_align/params CKPT_DIR=checkpoints bash scripts/refine_policy.sh
```

The envelope file is a synthetic module example, not a calibrated hardware limit.
Preparation applies one gripper conversion to state, applied action, requested command
and every behavior-proposal action row. Declare `gripper_encoding` in input metadata;
use `--no-flip-gripper` for an already closed-is-one corpus. Double conversion is rejected.
For accumulated rounds, use repeated critic `--corpus` arguments and a policy
`--round-manifest` JSON containing `corpora: [{dataset, credit_tag}, ...]`.
All features use the parent policy's normalization. The outcome-only reward mode
is available through `python -m facet0.refinement.reward --sparse`.
Missing behavior proposals cannot be used for critic regression. `encode --value-only`
extracts policy proposals for value scoring without behavior supervision.

## Serving and adaptation

```bash
MODEL_DIR=/path/to/pi_ref OPENPI_SRC=/path/to/openpi/src bash scripts/serve.sh
```

The policy API consumes state13, causal wrench history10×6, three RGB views and an
instruction. It returns executable Cartesian actions50×7 and predicted wrench50×6.
`contact_features(obs)` returns two fused contact tokens and their validity mask.
Checkpoint manifests specify model, normalization, feature and time conventions.

See [online adaptation](../facet-online-adaptation/README.md) for token training and local adaptation.
The shared wire uses 1=open; the policy dataset representation uses 1=closed.

## Tests

```bash
JAX_PLATFORMS=cpu pytest -q tests
pytest -q facet-online-adaptation/tests
```

Full-size weights, deployment datasets, judge models, a compliant robot controller
and hardware evaluation are supplied independently. `facet0.policy.hierarchy` loads
an aligned coarse checkpoint and a value-refined checkpoint with matching model,
normalization, tokenizer and cadence contracts. It schedules coarse refreshes at
5 Hz and contact-triggered refinement at 20 Hz; these rates are scheduling targets
and require a separately measured inference budget on deployment hardware.

The adaptation server accepts `--coarse-checkpoint`, `--wrench-envelope` (a JSON
array of six physical-axis limits), `--coarse-hz` and `--refinement-hz`. Configure
the robot runtime at 20 Hz to request refinement at that rate. Measured wrench
selects the contact branch, or the caller can supply a current boolean
`contact_phase`. The refined representation remains the bottleneck source.
Episode reset clears both cached chunks through the policy wire protocol.

The complete CPU example is documented in [examples](../examples/README.md).

## License

Apache-2.0. OpenPI and external model weights retain their respective notices and terms.
