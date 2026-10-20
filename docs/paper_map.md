# Components

| Module | Function |
| --- | --- |
| `pretraining` | Alternating joint flow and VQA optimization |
| `model`, `action_wrench` | State/history conditioning and next-step joint targets |
| `refinement/reward` | Judge-based completion, timing and wrench penalties |
| `refinement/critic` | Distributional action–wrench values and auxiliary supervision |
| `refinement/credit` | Short-horizon credit and regime budgets |
| `refinement/objective` | Contact-weighted conditional and unconditional flow learning |
| `policy` | Checkpoint loading, guidance, contact features and serving |
| `policy/hierarchy` | Coarse joint proposals and contact-triggered refinement scheduling |
| `facet-online-adaptation` | Frozen token encoding and bounded local actor–critic learning |

# Documented deviations

| Area | Paper | Code default | Reason | How to reproduce the paper setting |
| --- | --- | --- | --- | --- |
| Online Critic target | TD bootstrap from the target Actor with twin-Q minimum (Alg. 3, line 10) | `critic_target: mc`, regression on the discounted return of applied commands (behaviour value, one-step improvement) | A TD Critic of an Actor that has not yet completed the task holds value only at the labelled terminal step, so its action gradient carries no task signal | `learner.critic_target: td`; the twin-minimum TD path is unchanged |
| Online Actor schedule | TD3+BC Actor updates throughout adaptation | Critic-gated: BC seed held until held-out Critic checks pass; BC weight annealed on the RL-active Critic clock; per-window trust region | Q gradients from an unready Critic moved the Actor away from working behaviour | `learner.critic_gate: false` (ungated) |
