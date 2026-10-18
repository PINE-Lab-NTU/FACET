# Small-data end-to-end validation

Run from the bundle root with the canonical FACET environment (Python 3.11,
JAX 0.7.2, Flax 0.10.2, PyTorch 2.7.1, LeRobot 0.3.3). Install OpenPI at the
commit in [the setup guide](../docs/getting_started.md); do not put the historical pretraining snapshot's
`src/openpi` ahead of that checkout on `PYTHONPATH`.

```bash
export PYTHONPATH=/path/to/openpi/src:/path/to/openpi/packages/openpi-client/src
JAX_PLATFORMS=cpu python examples/small_pipeline.py --output-dir /tmp/facet-small-new
```

Use a new output directory. The example requires no model/data download after the
environment is installed. It trains a local SentencePiece tokenizer and a real
small SigLIP (`mu/32`) plus dummy-width Gemma network. All trainable parameters
participate in the relevant optimization; no output-only training filter or mocked
training loss is used. The production model's default vision variant is unchanged.

The command performs these stages:

1. Generate 3 synthetic LeRobot episodes, 36 frames with 3 RGB views, state, actions,
   causal force signals and 6 VQA records. Split by source demonstration into
   2 training episodes and 1 validation episode; compute training-only normalization.
2. Run 8 action–wrench and 2 VQA optimizer updates with independent optimizer state.
   Record parameter update norms and evaluate held-out AW/VQA losses.
3. Collect 24 transitions from the trained policy in a mock environment. Record
   policy-sampled joint proposals before outcomes, then convert all gripper fields
   consistently. Contact responses and judge labels are explicitly synthetic.
4. Encode actual policy representations and proposals, compute rewards, train a
   distributional critic for 20 updates, score values and assign regime-specific credit.
5. Run 2 value-guided refinement updates over the small full corpus so positive
   conditioning is exercised. Save and reload the refined checkpoint.
6. Record base-policy episodes, extract real frozen contact tokens and train the
   reconstruction bottleneck for 10 updates.
7. Collect successful/failed corrections, train BC, resume the same learner in RL,
   then reload in evaluation mode. Assert parameter changes in training, full wrench
   supervision coverage, command-record equality and frozen evaluation.
8. Load the trained coarse and refined policies through the hierarchy, exercise
   both contact regimes and verify the frozen token contract.
9. Check optimizer branch updates, positive refinement, frozen VL weights (allowing
   the documented bfloat16 storage conversion), shared normalization/tokenizer hashes,
   changed refinement parameters and checkpoint ancestry.

Outputs include `summary.json`, `checks.json`, per-stage logs, raw/prepared datasets,
configs, normalization, critic, credit and policy/token/local checkpoints. The
in-process transport adapter calls the actual frozen policy server; only network
transport and the physical robot are replaced. Backend unit tests separately check
HTTP/RTDE adapters and streaming command identity.

This is a computation and integration test, not a convergence, 20 Hz wall-clock
latency, robot-safety or task-success benchmark. A learned VLM judge, real data,
full-size weights and a compliant hardware controller remain external inputs.
