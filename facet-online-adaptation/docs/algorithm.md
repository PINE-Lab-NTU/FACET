# Learning

The state contains a frozen token and raw proprioception. Fixed normalization is
applied before separate token, kinematic, wrench and reference projections.
The bounded Actor outputs seven control values and six auxiliary wrench values.
The reference is conditioning information. The task action mask identifies active axes.

## Critic

Local critics evaluate applied commands with the same reference and state. TD targets
use a separate target Actor and twin target critics. Every replay row stores the
discounted return of its executed continuation. `critic_target` selects the regression
target:

- `mc` (default): regress Q(s, a_applied) on that return, the value of the behaviour that
  actually ran.
- `td`: bootstrap with the target Actor's next action.
- `td_mc_max`: the TD target, lower-bounded by the return.

Rows without a known return keep the TD target. The default is `mc` because a TD Critic
of an Actor that has not yet completed the task carries value only at the labelled
terminal step, so its action gradient holds no task signal.

Critic training retains both successful and failed episodes. Unknown outcomes are recorded
without assigning terminal success or failure rewards.

## Actor schedule (critic-gated TD3+BC)

1. **`bc` mode builds the seed.** Each episode that adds newly confirmed successful
   corrections triggers `intervention_updates_per_episode` BC steps at `bc_lr`. Autonomous
   episodes add no BC steps.
2. **`rl` mode holds the seed while the Critic trains.** The Actor at the first RL update
   becomes the BC seed. Until the readiness gate opens, Critic updates run and the Actor is
   held (`hold_actor_until_critic_ready`).
3. **The gate opens on held-out monitor episodes.** Every `monitor_every`-th labelled
   episode is excluded from all training. Every `gate_eval_every` Critic updates, the gate
   evaluates the monitor and requires all of the following:
   - at least `gate_min_updates` Critic updates;
   - at least `gate_min_terminals` successful and failed terminals;
   - terminal Brier score ≤ 0.1;
   - mean success-terminal Q ≥ 0.7 and |mean failure-terminal Q| ≤ 0.3;
   - mean twin-Q disagreement ≤ 0.1;
   - RMS change of Q and target Q on the recorded monitor actions ≤ 0.02;
   - p95 xyz drift of the Actor from the seed ≤ 1 mm;
   - successful-behaviour BC error ≤ 1.1× the seed's.

   The gate opens after `gate_consecutive_passes` fully passing evaluations. Any failure
   closes it, and reopening requires a fresh streak. A change of monitor population counts
   as one unstable evaluation.
4. **While the gate is open, every Actor step is TD3+BC:**
   `actor_q_weight * (-Q1) + beta * BC + wrench_prediction_weight * wrench`.
   - The Q term uses the online half of the 50/50 batch.
   - BC regresses the demo half onto its applied actions (`td3_bc_target: demo`).
   - `beta` anneals linearly from `bc_beta` to `bc_beta_end` over `bc_anneal_critic_updates`
     Critic updates taken while RL is active. Paused periods do not advance this clock.
   - Confirmed corrections reach the Actor through this BC term. Separate intervention-BC
     steps in `rl` mode are off by default (`rl_intervention_updates_per_episode: 0`); when
     enabled they outnumber the Q steps.
5. **Matched control.** `actor_q_weight: 0` keeps every schedule, batch and BC step and
   removes only the Q gradient. Compare it with the TD3+BC Actor by frozen `eval` runs to
   isolate the RL contribution.

The wrench head shares the actor trunk and additionally reads the normalized executable
command. At inference it predicts the consequence of the safety-filtered command; in replay
it reads the recorded command. All finite replay rows therefore supervise the head without
an equality test against the current actor. `wrench_coverage` reports the supervised fraction.
Critics still evaluate only 7-D actions.

The gate is a conservative monitor, not evidence of task success. Success rates require
frozen evaluation on the robot.

Every episode logs `episode_bc_updates`, `episode_td3_actor_updates`,
`episode_q_gradient_share`, `gate_active` and `rl_active_critic_updates`. Gate evaluations
log every check value and the names of failed checks.
