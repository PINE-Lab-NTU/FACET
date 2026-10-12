# Paper-to-code map

Facet-0 (arXiv:2609.01596) has three method modules that share the semantic–contact
representation `h_t^c` (Table 6). This repository implements **value-guided policy
refinement** (Sec. 5.1–5.2, Alg. 2) together with the policy model it refines. The
remaining parts were run on separate servers and are not in this repository; they enter
here only through their outputs: the aligned checkpoint `pi_align`, the deployment
rollouts, and the judge's sub-goal and phase columns.

## In this repository

| Paper | Element | Code |
| --- | --- | --- |
| Eq. (1) | 13-D state `s_t = [x_t; g_t; w_t]` | `facet0/conventions.py` |
| Sec. 4.1 | three views and instruction in the prefix; `E_s(s^kin)` and `E_w(W_{t-K+1:t})`, K = 10, only in the action expert; fusion `F_fuse` by the expert's attention | `facet0/model.py` |
| Sec. 4.1 | joint proposal `Y_t in R^{H x 13}`, H = 50, only `A` executable, `W^` in physical units after reversing normalization | `facet0/model.py`, `facet0/policy/transforms.py` |
| Sec. 4.2, App. D.1 | row k pairs `a_{t+k}` with `w_{t+k+1}`; masks `M^a`, `M^w` | `facet0/action_wrench.py`, `facet0/data/frames.py` |
| Eq. (3) | action–wrench flow matching with per-slice mask normalization, `lambda_pre` = 0.1, tau ~ Beta(1.5, 1) | `facet0/model.py` (`flow_matching_errors`), `facet0/refinement/objective.py` |
| Sec. 5.1, Eq. (4) | interaction reward from sub-goal completion, time budget and per-phase wrench envelope | `facet0/refinement/reward.py` |
| Sec. 5.1, Eq. (5) | Action–Wrench Critic `Z_psi(h^c, Y)`: categorical return distribution, distributional TD with target parameters `psi_bar`, successor proposal `Y_{t+1} ~ pi_theta` | `facet0/refinement/critic.py` |
| Sec. 5.1 | `V_psi(h^c) = E_{Y ~ pi_theta}[Q_psi(h^c, Y)]` | `facet0/refinement/critic.py` (`value`), `facet0/refinement/encode.py` |
| App. D.2 | progress-calibrated targets; four auxiliary heads (near-future wrench, contact intensity, within-contact progress, success ranking) | `facet0/refinement/critic.py` |
| App. D.2 | contact intensity `c_t` from causally smoothed wrench differences; regime `rho_t` by thresholding `c_t` | `facet0/contact.py` |
| Sec. 5.2, Eq. (6) | undiscounted short-horizon credit `delta_t^(N)`, N < H | `facet0/refinement/credit.py` |
| Sec. 5.2, App. D.2 | top-ranked fraction within each regime at a matched overall positive rate; label `b_t` | `facet0/refinement/credit.py` |
| Sec. 5.2, Eq. (7) | `E_t[(1 + lambda c_t) L_FM]`; all frames train the unconditional branch, `b_t = 1` frames also the conditional branch | `facet0/refinement/objective.py` |
| App. D.2 | prompt-level conditioning (tag on the instruction); classifier-free-guidance sampling, `s = 0` exactly unconditional | `facet0/policy/guidance.py` |
| Table 6 | refinement trains `pi_ref` (action expert, `E_s`, `E_w`) and `Q^AW_psi`; the VL backbone is frozen | `facet0/model.py` (`vl_backbone_filter`), `facet0/refinement/train.py` |
| Alg. 2, lines 1–9 | one round: rewards, critic on the accumulated rollouts D, credit, contact-validity-masked `Y^data`, flow-matching update of `pi_ref` | `scripts/prepare_rollouts.sh` … `scripts/refine_policy.sh` |

## Run on separate servers (not in this repository)

| Paper | Element |
| --- | --- |
| Sec. 3, App. B | ManuFacet-1K collection and curation: multi-center acquisition, shared-clock 15 Hz timeline, segment-level validity extraction, saliency-based chunk sampling, VLM sub-task / next-instruction QA, seven-phase annotation, anomaly reflow |
| Sec. 4, Alg. 1 | semantic–contact alignment: VQA branch, structured attention mask with the VQA path, 4:1 optimizer alternation, mixture pyramid; produces `pi_align` |
| Sec. 5.1 | the VLM judge `Omega` that segments rollouts into sub-goals and contact phases (its per-frame output feeds Eq. 4 here) |
| Sec. 4.1, App. C | 20 Hz refinement expert and its guidance by `Q_psi`, 200 Hz compliant controller, safety operator `S_task` |
| Alg. 2, line 3 | collecting rollouts of `pi_ref` and supervisor interventions on the robot cells |
| Alg. 2, line 10; Sec. 5.3; Alg. 3; App. D.3 | FACET bottleneck encoder `E_z` and local policy adaptation (bounded TD3 actor, twin critics, intervention anchor, auxiliary next-wrench head) |
| Sec. 6, App. A, E, F | evaluation suite, protocol and baselines |

Values the paper leaves open are listed in [implementation_notes.md](implementation_notes.md).
