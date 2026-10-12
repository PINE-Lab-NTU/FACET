"""State, action and proposal layout (Eq. 1, Sec. 4.1) shared by every stage."""

from __future__ import annotations

STATE_DIM = 13
KIN_DIM = 7
KIN_SLICE = slice(0, KIN_DIM)
GRIPPER_INDEX = 6
WRENCH_SLICE = slice(7, 13)
WRENCH_DIM = 6
WRENCH_WINDOW = 10  # K

ACTION_DIM = 7
ACTION_HORIZON = 50  # H
PROPOSAL_DIM = ACTION_DIM + WRENCH_DIM
PROPOSAL_ACTION_COLUMNS = slice(0, ACTION_DIM)
PROPOSAL_WRENCH_COLUMNS = slice(ACTION_DIM, PROPOSAL_DIM)

# The first six action dimensions are trained as deltas from the current pose; the gripper stays absolute.
DELTA_ACTION_DIMS = 6

# Three synchronized RGB views I_t^{1:3}: LeRobot feature -> pi0.5 image slot.
CAMERAS = {
    "observation.images.view1": "base_0_rgb",
    "observation.images.hand": "left_wrist_0_rgb",
    "observation.images.view2": "right_wrist_0_rgb",
}

# Positive-interaction label b_t reaches the policy as a literal suffix on the
# instruction (App. D.2, prompt-level conditioning).
GUIDANCE_TAG = "\nAdvantage: positive"
