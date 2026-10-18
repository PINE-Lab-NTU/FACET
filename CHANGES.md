# Changes

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
