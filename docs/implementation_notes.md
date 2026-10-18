# Interfaces

The policy uses a 7-D kinematic token and a causal 10×6 wrench token. The two
fused outputs define contact features. Joint targets contain 7 action and 6
next-step wrench columns, with separate validity masks and element-mean losses.

Pretraining alternates two optimizers. Post-training freezes the visual-language
backbone and uses distributional return learning, contact-selective labels and
weighted unconditional/positive flow objectives. Current critic inputs are recorded
behavior proposals; successor and value inputs are current-policy samples.

Checkpoint manifests include model settings, parameter and normalization checksums,
parent identity and temporal conventions. The normalization remains fixed across
refinement rounds and downstream token extraction.
