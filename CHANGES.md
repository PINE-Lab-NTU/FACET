# Changes

## Online adaptation: critic-gated TD3+BC

| Change | Reason | Interface | Validation |
| --- | --- | --- | --- |
| Hold the BC seed in `rl` mode until a Critic readiness gate passes on held-out monitor episodes (min updates, terminal coverage/Brier/success-Q/failure-Q, twin-Q disagreement, fixed-probe Q and target-Q stability, seed drift ≤ 1 mm, BC degradation ≤ 1.1×; 3 consecutive passes; any failure pauses RL) | Q gradients from an uncalibrated Critic moved the Actor away from working behaviour | `learner.critic_gate`, `hold_actor_until_critic_ready`, `monitor_every`, `gate_*` | Gate unit tests; held-Actor bitwise test |
| Anneal the TD3+BC BC weight 1.0 → 0.3 over 3000 Critic updates counted only while RL is active | BC weight should fall only as the Critic earns trust | `bc_beta`, `bc_beta_end`, `bc_anneal_critic_updates` | Schedule and pause-clock tests |
| Default Critic target `mc` (discounted return of applied commands) | A TD Critic of an Actor that never completes the task carries value only at the labelled terminal | `critic_target` (`td`, `td_mc_max` kept) | Existing target tests |
| Run intervention BC only after episodes that add confirmed labels; no separate intervention BC in `rl` mode by default | Replaying all labels after every episode made BC ≈ 93% of Actor updates | `intervention_updates_per_episode`, `rl_intervention_updates_per_episode` | Label-gated BC test |
| Add `actor_q_weight` (0 = matched BC control with identical schedule and batches) | Isolate the RL contribution in frozen evaluation | `actor_q_weight` | Bitwise Critic-independence test |
| First Actor step at Critic update `critic_updates_per_actor`, not update 0 | The Actor was stepped on an untrained Critic | — | Updated cadence tests |
| Wire `demo_success_only` and `demo_seed_success_episodes` | Both were read nowhere | config | Demo-rule tests |
| Optional cosine Critic LR decay; per-episode BC/Q-gradient update counts; checkpoint format v4 (v3 migrates) | Auditable RL contribution and resumable gate state | `critic_lr_schedule`; metrics; `facet_online_adaptation_v4` | Round-trip and migration tests |

The schedule follows our offline critic-gated TD3+BC procedure and the update cadence of
our internal robot runs; those logs are not part of this repository. The gate does not
establish robot success; use frozen `eval` runs, including the `actor_q_weight: 0` control,
for that. The `mc` default deviates from the paper's TD target and is listed in
`docs/paper_map.md`.

Review follow-up:

| Change | Reason | Validation |
| --- | --- | --- |
| Actor-side gate failures no longer freeze the Actor. A trust-region failure re-anchors at the current Actor and needs a fresh Critic streak; a behaviour failure runs BC-only recovery steps. Critic-side failures still hold the Actor. | With hold enabled, a drift or behaviour failure froze the Actor, so the failed quantity could never change and RL stopped permanently | Re-anchor/reopen test with the Actor pushed far outside the region; BC-recovery test; Critic-hold test |
| Monitor hold-out only in `rl` mode with the gate enabled | `bc` mode discarded 20% of corrections for a monitor it does not use | bc-mode and ungated retention tests |

## 2026-10-18

This date identifies the present correction, not the development date of inherited source.

| Change | Reason | Paper / interface | Validation |
| --- | --- | --- | --- |
| Convert all action-bearing gripper fields together; reject double conversion and invalid vectors | Prevent behavior proposals and executed actions from describing opposite gripper states | Sec. 4.1, 5.1; recorded proposal/request contract | Flat/nested proposal regression checks and actual rollout→encode example |
| Condition the auxiliary wrench head on the applied action; train it on every replay row | Current-policy equality masking discarded old-policy and human transitions | Eq. 9, Appendix D.3; engineering realization of action–contact coupling | Mismatched-intervention regression, nonzero wrench loss, full replay coverage |
| Query the bounded actor proposal in the actor Q objective | Keep proposal optimization distinct from critic regression on applied commands | Sec. 5.3 | Actor/critic tests and BC→RL example |
| Synchronously send and return the final endpoint; heartbeat repeats it or stops on drift | Eliminate a second, unrecorded motion projection | Appendix D.3 action/execution separation | Driver-send equality and state-drift regression checks |
| Carry the streaming execution period into learner projection and recording | Make execution and replay timing explicit | Fixed task safety operator | Runtime and streaming tests |
| Add coarse/refinement policy scheduling with checkpoint compatibility checks and episode reset | Supply the software path for two-frequency joint proposals | Sec. 4.1, Appendix C | 5/20 Hz schedule tests and actual trained-checkpoint hierarchy inference; no latency claim |
| Preserve model configuration and optional local tokenizer through encoding, refinement and serving | Avoid silently replacing small/nondefault checkpoint settings; permit an offline example | Shared representation/checkpoint contract | Tokenizer checksum, parent identity and reload checks |
| Add an explicit small SigLIP option; retain the original production default | Permit all-parameter CPU validation without replacing training functions | Same joint-flow and causal VQA computation at smaller scale | Full AW/VQA updates with measured parameter changes |
| Correct Algorithm 2 line reference to line 10; save completed update numbers | Remove stale alignment comment and checkpoint step ambiguity | Eq. 7, Algorithm 2 line 10 | Checkpoint manager and example step counts |

The action-conditioned wrench implementation adds an internal action input to the
consequence head. This is an explicit engineering choice to supervise off-policy
transitions correctly, not a claim of bitwise identity with the unreleased trained
network. The local observation interface remains token + proprioception + reference,
and the executable action remains seven-dimensional.

The source includes a reproducible synthetic-data pipeline. It does not contain the
ManuFacet-1K corpus, original trained weights, a learned VLM judge, or a compliant
physical controller. Synthetic labels and tiny models validate computation and
contracts; they do not reproduce the paper's hardware metrics.
