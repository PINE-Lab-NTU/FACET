# Facet Online Adaptation

Bounded absolute Cartesian actor–critic learning from a frozen Facet policy,
contact bottleneck tokens, measured robot state and reference actions.

## Setup

```bash
pip install -e '.[test]'
python -m pytest -q
python -m facet_adaptation.cli demo --output-dir /tmp/facet-demo --episodes 3 --steps 12
```

UR RTDE and Franka HTTP provide measured-state, command and hold interfaces.
Install optional `ur`, `cameras` and `teleop` dependencies for the corresponding devices.
A compliant hardware controller runs independently.

## Workflow

1. Start `facet_adaptation.integrations.serve --backend facet0 --checkpoint PATH --emit-prefix`
   in the canonical FACET policy environment.
2. Collect base-policy episodes with recorded RGB images and causal wrench histories.
3. Use `facet-encoder extract`, then `facet-encoder train`, to train a contact-token bottleneck.
4. Serve the same policy with `--token-checkpoint`. Keep the printed policy and token IDs.
5. Copy `configs/module.example.yaml`, select the robot backend and configure the
   calibrated action box, workspace, camera roles and frozen model IDs.
6. Collect successful corrections in `bc` mode, continue in `rl`, then evaluate in `eval`.

The module example uses the mock backend and synthetic bounds. `ur.example.yaml`
and `franka.example.yaml` describe hardware connections without task geometry.
Hardware configurations must provide native-frame workspace bounds and a task action box.

The Actor uses four modality projections and produces an absolute action and a
non-commanded next-wrench prediction conditioned on the applied command. Only
applied 7-D commands enter critic regression; actor updates query the bounded proposal.
Target Actor and twin target critics use Polyak updates. BC uses confirmed successful
interventions. Reference actions are retained separately from human commands.

Commands are `resume`, `pause`, `teleop`, `policy`, `confirm`, `success`, `failure`,
and `stop`. `confirm` labels the latest contiguous human segment as successful.
Evaluation disables teleoperation and optimizer updates. Every episode is recorded,
including interrupted episodes without an outcome label.

State is `[pose6, opening, wrench6]` with 1=open. Wrench history is sampled on a
15 Hz target grid from timestamped causal measurements. The current observation,
reference, requested command and accepted endpoint have distinct meanings.
The streaming adapter sends the final filtered endpoint synchronously and returns
that exact command. Its heartbeat repeats the same endpoint. State drift requiring
a different endpoint stops streaming instead of introducing an unrecorded action.
The decision uses the smaller of runtime and streaming periods for rate limiting;
the learner uses that same `execution_dt`. Lower runtime frequency can consequently
reduce travel speed. Verify task timing after configuring the control rates.

New checkpoints use `facet_online_adaptation_v3`. They contain Actor, target Actor,
critics, target critics, optimizer states, replay, normalization and RNG state.
Older checkpoint schemas require an explicit conversion or a fresh run.
The v3 action-conditioned wrench head changes the parameter tree; v2 learning state
is deliberately rejected rather than silently resumed with a random prediction head.
