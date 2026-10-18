# Learning

The state contains a frozen token and raw proprioception. Fixed normalization is
applied before separate token, kinematic, wrench and reference projections.
The bounded Actor outputs seven control values and six auxiliary wrench values.
The reference is conditioning information. The task action mask identifies active axes.

Local critics evaluate applied commands with the same reference and state. TD targets
use a separate target Actor and twin target critics. The Actor objective combines Q1,
successful intervention BC and next-wrench regression conditioned on the applied
replay action. The wrench head shares the actor trunk and additionally reads the
normalized executable command. At inference it predicts the consequence of the
safety-filtered command; in replay it reads the recorded command. All finite replay
rows therefore supervise the head without an equality test against the current actor.
`wrench_coverage` reports the supervised fraction. Critics still evaluate only 7-D actions.
Critic training retains both successful and failed episodes. Unknown outcomes are recorded
without assigning terminal success or failure rewards.
