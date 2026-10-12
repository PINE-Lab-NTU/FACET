# Facet-0 post-training

Value-guided policy refinement for [Facet-0: A Robotic Foundation Model for Contact-Rich
Precise Manipulation](https://arxiv.org/abs/2609.01596) (Sec. 5.1–5.2, Alg. 2).

The policy pairs a PaliGemma vision–language backbone with a flow-matching action expert.
The kinematic state and a causal wrench history (K = 10) enter only the action expert,
which generates each H = 50 action chunk together with the wrist wrench it is expected
to induce. Only the action is executed. Deployment rollouts are segmented into
sub-goals and rewarded (Eq. 4). An **Action–Wrench Critic** learns the return
distribution of the policy's joint proposal from the semantic–contact representation
(Eq. 5). **Contact-selective credit** (Eq. 6) ranks frames within their contact regime
to label the decisive interactions, and a **contact-weighted flow-matching** update
(Eq. 7) refines the shared action expert while the vision–language backbone stays
frozen. At deployment a guidance scale trades between the unconditional and the
label-conditioned branch.

## Scope

This repository contains the refinement round: rollout ingestion, interaction reward,
encoding of rollouts with the current policy, the Action–Wrench Critic, contact
intensity and credit, the refinement objective and trainer, the Facet-0 policy model
and its deployment.

These parts of Facet-0 were run on separate servers and are not included:

* ManuFacet-1K collection and curation (Sec. 3, App. B);
* semantic–contact alignment, which trains the aligned policy `pi_align` that the first
  refinement round starts from (Sec. 4, Alg. 1);
* the VLM judge that segments rollouts into sub-goals and phases for Eq. (4);
* the 20 Hz refinement expert, 200 Hz compliant controller and safety operator (App. C);
* the FACET bottleneck encoder and local policy adaptation (Sec. 5.3, Alg. 3);
* evaluation and baselines (Sec. 6).

[docs/paper_map.md](docs/paper_map.md) maps every section, equation and algorithm line to
code or to "separate servers".

## Layout

```
facet0/
  conventions.py        state / action / proposal layout (Eq. 1), H = 50, K = 10, guidance tag
  model.py              Facet-0 policy: E_s and E_w tokens in the action expert, Y in R^{H x 13}
  contact.py            contact intensity c_t and regime rho_t
  action_wrench.py      Y^data wrench columns (next-step pairing), masks M^a / M^w, history W
  data/                 rollout ingestion, corpus frames, normalization statistics
  refinement/
    reward.py           interaction reward (Eq. 4) and returns
    encode.py           h^c and proposals Y ~ pi_ref for every stored frame
    critic.py           Action-Wrench Critic (Eq. 5) with four auxiliary heads
    credit.py           contact-selective credit (Eq. 6) and labels b_t
    objective.py        action-wrench loss (Eq. 3) and refinement loss (Eq. 7)
    data.py, train.py   refinement data pipeline and trainer (Alg. 2, line 8)
  policy/               transforms, guided sampling, checkpoint I/O, policy, server
scripts/                one refinement round, step by step (see docs/recipe.md)
docs/                   paper map, recipe, implementation notes
tests/                  CPU tests, including dummy-size model tests
```

## Install

```bash
git clone -b post-training https://github.com/PINE-Lab-NTU/FACET.git
git clone https://github.com/Physical-Intelligence/openpi.git   # JAX pi0.5 stack
git -C openpi checkout 650c5b0
pip install -e openpi -e openpi/packages/openpi-client
pip install -e "./FACET[policy,test]"
```

Developed and tested with openpi at commit `650c5b0`, Python 3.11, JAX 0.7.2, flax 0.10.2,
optax 0.2.4, orbax-checkpoint 0.11 and lerobot 0.3.3.

## Run a refinement round

```bash
export OPENPI_SRC=/path/to/openpi/src DATASET=/data/rollouts_r1 ROUND=r1
RAW=/data/raw_rollouts_r1 bash scripts/prepare_rollouts.sh
bash scripts/rewards.sh
PI_REF=<pi_align>/params bash scripts/encode.sh
CRITIC=critics/r1 bash scripts/train_critic.sh
CRITIC=critics/r1 bash scripts/credit.sh
PI_REF=<pi_align>/params CKPT_DIR=/nvme/ckpts bash scripts/refine_policy.sh
```

[docs/recipe.md](docs/recipe.md) covers later rounds and what to check after each step.

## Use the released policy

The `facet0-post-training` weights in [Pinelab/Facet-0](https://huggingface.co/Pinelab/Facet-0)
are the policy after one refinement round with this code.

```bash
hf download Pinelab/Facet-0 --include "facet0-post-training/*" --local-dir .
```

```python
import facet0.policy

policy = facet0.policy.load("facet0-post-training")            # or a checkpoint step of your own round
history = facet0.policy.WrenchHistory()                        # latest 10 wrenches, 15 Hz timeline
history.push(state[7:13])
out = policy.infer({
    "observation/state": state,                                # float32 [13]: pose, gripper, wrench
    "observation/wrench_history": history.window(),            # float32 [10, 6], oldest first
    "observation/image": view1,
    "observation/wrist_image": hand,
    "observation/extra_view_image": view2,
    "prompt": "press down the RAM until it locks into place",
})
out["actions"]   # [50, 7] absolute pose + gripper
out["wrench"]    # [50, 6] anticipated wrench (N, N m)
```

`MODEL_DIR=facet0-post-training OPENPI_SRC=... bash scripts/serve.sh` serves it over the
openpi websocket protocol.

## Tests

```bash
PYTHONPATH=$OPENPI_SRC:. pytest tests
```

## Citation

```bibtex
@article{deng2026facet0,
  title   = {Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation},
  author  = {Deng, Haoyuan and Liu, Haichao and Guo, Wenkai and Ling, Yuan and Yang, Zaijia and
             Xue, Yuanjiang and Sun, Haosheng and Wang, Liangzi and Wang, Ziwei},
  journal = {arXiv preprint arXiv:2609.01596},
  year    = {2026}
}
```

## License

Apache-2.0. The model and trainer build on openpi (Apache-2.0).
