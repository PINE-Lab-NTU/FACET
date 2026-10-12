# Implementation notes

The code follows the paper's formulation. This page records the concrete values and
realizations for details the paper leaves open, and how the code was verified.

## Policy model (`facet0/model.py`)

* Backbone: openpi pi0.5 with PaliGemma (`gemma_2b`) and a `gemma_300m` action expert,
  adaRMS time conditioning, tau ~ Beta(1.5, 1) rescaled to [0.001, 1].
* `E_s` is a linear map of `s^kin` (7) to one expert token. `E_w` is a two-layer MLP
  (swish) of the flattened window `W_{t-9:t}` (10 x 6) to one expert token. Both tokens
  form one attention block after the prefix and before the chunk. The expert's
  attention to the prefix plays the role of `F_fuse`.
* `h_t^c` is the expert output at the two tokens, concatenated (2 x 1024), taken under
  the untagged instruction at tau = 1.
* The state vector and the 13 target columns use per-axis quantile normalization. The
  first six action dimensions are deltas from the current pose, and the gripper is
  absolute. Un-normalizing the decoded columns [7:13] gives `W^` in N and N m.
* The wrench targets are the causal 5-frame moving average of the measured wrench.

## Refinement objective (`facet0/refinement/objective.py`, `train.py`)

* `lambda_pre` = 0.1, the value of Alg. 1, reused for Eq. (3) inside Eq. (7). `lambda` = 1.
* Each step runs two forward passes: an untagged pass for every frame and a tagged pass
  that contributes only for `b_t = 1`.
* Frozen: SigLIP and the PaliGemma language expert, including token embeddings.
  Trained: the action expert, the action input and output projections, the time MLP,
  `E_s` and `E_w`.
* AdamW, gradient clip 1.0, warm-up of 2,000 steps to 1e-5, EMA 0.999. Checkpoints are
  saved synchronously and carry `assets/norm_stats.json`.

## Reward (`facet0/refinement/reward.py`)

* Judge output is read from the columns `subgoal_index`, `subgoal_complete` (`Omega_t`)
  and `phase` (`phi_t`). A rollout without them is one sub-goal that completes on its
  last frame if the deployment succeeded.
* `Delta_t` counts the frames since the current sub-goal started, inclusive, and `T_max`
  is given in frames. `||w_t||_phi` is the largest per-axis ratio `|w_i| / envelope_i`.
* `alpha`, `beta`, `T_max` and the envelopes are configuration. They default to zero or
  off, in which case `r_t = Omega_t`.

## Action–Wrench Critic (`facet0/refinement/critic.py`, `encode.py`)

* Inputs: `h^c` (fixed for the round) and the normalized proposal `Y`, each through
  LayerNorm and a linear map to 512, followed by a two-layer GELU trunk of width 512.
* The return distribution uses 201 atoms on [-1, 1] with gamma = 0.99. `D_dist` is the
  cross-entropy to the categorical projection of `r_t + gamma (1 - d_t) Z_psi_bar`.
  `psi_bar` is a Polyak average with rate 0.005.
* Progress calibration regresses `E[Z]` onto `t / (T - 1)` on successful rollouts, with
  weight 1.
* The auxiliary heads:

  | Head | Target | Loss | Weight |
  | --- | --- | --- | --- |
  | near-future wrench | normalized filtered wrench 8 frames ahead | Huber | 0.05 |
  | contact intensity | `c_t` | binary cross-entropy | 0.2 |
  | within-contact progress | `t / (T - 1)` in the contact regime of successful rollouts | MSE | 0.2 |
  | success ranking | successful contact frames above failed ones | pairwise hinge, margin 0.1 | 0.2 |

* Training: AdamW 3e-4, batch 256, 20,000 steps. `V_psi` averages `Q_psi` over the M
  stored proposals of a frame (M = 1 by default). Proposals are sampled with guidance
  scale 1 and 10 integration steps.
* Every round re-encodes all corpora in D with the current `pi_ref` and retrains the
  critic on their union.

## Contact and credit (`facet0/contact.py`, `facet0/refinement/credit.py`)

* `c_t`: causal filter of 5 frames, lag 8. The per-episode scale is the 0.8-quantile of
  `d_t`, floored by the corpus median. `rho_t` means contact when `c_t` saturates (>= 1).
* N = 20 frames (1.3 s on the 15 Hz timeline). The top 20% of frames are positive
  within each regime, so the overall positive rate is also 20%.

## Verification

* 19 CPU tests (`pytest tests`) cover the following:
  * causal filtering and the contact scale;
  * Eq. (4), including judge segmentation;
  * Eq. (6) and the regime budgets;
  * next-step pairing and masks;
  * the Eq. (3)/(7) composition;
  * the categorical projection and critic training;
  * model tests at dummy size:
    * the per-element flow-matching error matches openpi's `compute_loss`;
    * `h^c` is unaffected by the noised chunk and by the tag;
    * guidance scale 0 reproduces unconditional sampling;
    * the freeze filter covers exactly the vision–language backbone.
* An end-to-end run on a 2-episode, 40-frame slice of a real rollout corpus with a
  dummy-size model covered every stage. It computed normalization statistics
  (67 / 13 dims), rewards, and encoding (h^c and two proposals per frame). It then
  trained and scored the critic, computed credit, ran two refinement steps and saved a
  checkpoint. The checkpoint loaded with `facet0.policy.load`, which returned actions
  [50, 7] and wrench [50, 6].
* The released `facet0-post-training` weights are one full-size round of this code.
