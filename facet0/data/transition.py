"""Record behavior proposals and executed transition metadata."""

import numpy as np


def behavior_fields(
    proposal,
    requested,
    applied,
    *,
    policy_id,
    execution_map_id,
    proposal_time,
    command_time,
    observation_time,
    next_observation_time,
):
    proposal = np.asarray(proposal, np.float32)
    requested, applied = np.asarray(requested, np.float32), np.asarray(applied, np.float32)
    if proposal.shape != (50, 13) or requested.shape != (7,) or applied.shape != (7,):
        raise ValueError("invalid proposal or command shape")
    if not all(np.isfinite(v).all() for v in (proposal, requested, applied)):
        raise ValueError("nonfinite behavior record")
    if not np.allclose(proposal[0, :7], requested, atol=1e-5):
        raise ValueError("proposal/request mismatch")
    if (
        not policy_id
        or not execution_map_id
        or not observation_time <= proposal_time <= command_time < next_observation_time
    ):
        raise ValueError("invalid behavior identity or timestamps")
    return {
        "behavior_joint_proposal": proposal.tolist(),
        "requested_command": requested.tolist(),
        "action": applied.tolist(),
        "behavior_policy_id": policy_id,
        "execution_map_id": execution_map_id,
        "proposal_time": proposal_time,
        "command_time": command_time,
        "timestamp": observation_time,
        "next_observation_time": next_observation_time,
    }
