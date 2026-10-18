"""Conventions."""

from __future__ import annotations

STATE_DIM = 13
KIN_DIM = 7
KIN_SLICE = slice(0, KIN_DIM)
GRIPPER_INDEX = 6
WRENCH_SLICE = slice(7, 13)
WRENCH_DIM = 6
WRENCH_WINDOW = 10

ACTION_DIM = 7
ACTION_HORIZON = 50
PROPOSAL_DIM = ACTION_DIM + WRENCH_DIM
PROPOSAL_ACTION_COLUMNS = slice(0, ACTION_DIM)
PROPOSAL_WRENCH_COLUMNS = slice(ACTION_DIM, PROPOSAL_DIM)


DELTA_ACTION_DIMS = 6


CAMERAS = {
    "observation.images.view1": "base_0_rgb",
    "observation.images.hand": "left_wrist_0_rgb",
    "observation.images.view2": "right_wrist_0_rgb",
}


GUIDANCE_TAG = "\nAdvantage: positive"
